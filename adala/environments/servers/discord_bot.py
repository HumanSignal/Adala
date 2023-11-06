import os

import asyncio

import discord
import aiosqlite
from typing import List, Dict, Any
from discord.ext import commands
from discord.ui import View
from adala.environments.servers.base import BaseAPI, Prediction, GroundTruth, STORAGE_DB


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
CHANNEL_ID = 1170485562692603944


@bot.event
async def on_ready():
    print(f'Hooray!! Logged in as {bot.user.name}')


@bot.command()
async def hello(ctx):
    await ctx.send("Hello World")


@bot.event
async def on_message(message):
    if message.is_system() or message.author.bot:
        return

    if message.channel.type is discord.ChannelType.private:
        return

    if message.channel.id != CHANNEL_ID:
        return

    if message.type == discord.MessageType.reply:
        # reply in thread
        initial_message_id = message.reference.message_id
        print(f'Got reply in thread for message id: {initial_message_id}', message)
        async with aiosqlite.connect(STORAGE_DB) as db:
            async with db.execute('SELECT * FROM discord_gt_message WHERE message_id = ?', (initial_message_id,)) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    print(f'No ground truth message found for message id: {initial_message_id}')
                    return
                prediction_id, skill_name = int(row[1]), row[2]

            # update ground truth with reply
            await db.execute('UPDATE ground_truth SET gt_data = ? '
                             'WHERE prediction_id = ? AND skill_name = ?',
                             (message.content, prediction_id, skill_name))
            await db.commit()

    # Process other messages normally
    await bot.process_commands(message)


@bot.event
async def on_interaction(interaction: discord.Interaction):

    async def update_ground_truth_match(prediction_id: int, skill_name: str, match: bool):
        async with aiosqlite.connect(STORAGE_DB) as db:
            await db.execute('UPDATE ground_truth SET gt_match = ?, gt_data = NULL '
                             'WHERE prediction_id = ? AND skill_name = ?',
                             (match, prediction_id, skill_name))
            await db.commit()
            print(f'Updated ground truth for prediction id: {prediction_id} with match: {match}')

    if interaction.type == discord.InteractionType.component:
        await interaction.response.defer(ephemeral=True)

        custom_id = interaction.data['custom_id']
        action, prediction_id_str, skill_name = custom_id.split(':')
        prediction_id = int(prediction_id_str)  # Convert prediction_id to int

        if action == 'accept':
            # Handle the accept action
            await update_ground_truth_match(prediction_id, skill_name, True)
            # React with a checkmark emoji to the message
            await interaction.message.add_reaction('✅')
        elif action == 'reject':
            # Handle the reject action
            await update_ground_truth_match(prediction_id, skill_name, False)
            # React with a cross mark emoji to the message
            await interaction.message.add_reaction('❌')


class AcceptRejectView(View):

    def __init__(self, prediction_id: int, skill_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_item(discord.ui.Button(
            label='Accept',
            style=discord.ButtonStyle.success,
            custom_id=f'accept:{prediction_id}:{skill_name}'
        ))
        self.add_item(discord.ui.Button(
            label='Reject',
            style=discord.ButtonStyle.danger,
            custom_id=f'reject:{prediction_id}:{skill_name}'
        ))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        # Here you can add additional checks if needed
        return True


class DiscordAPI(BaseAPI):

    async def init_db_gt_message(self):
        async with aiosqlite.connect(STORAGE_DB) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS discord_gt_message (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    skill_name TEXT NOT NULL,
                    message_id INTEGER NOT NULL
                )
            ''')
            await db.commit()

    def start_discord_bot(self):
        asyncio.create_task(bot.start(DISCORD_TOKEN))

    def stop_discord_bot(self):
        asyncio.create_task(bot.close())

    async def request_feedback(
        self,
        predictions: List[Dict[str, Any]],
        skills: List[Dict[str, Any]],
        db: aiosqlite.Connection
    ):
        await bot.wait_until_ready()
        channel = bot.get_channel(CHANNEL_ID)
        if not channel:
            raise Exception(f'Channel with id {CHANNEL_ID} not found')
        ground_truths = []
        skill_names = [skill['name'] for skill in skills]
        for skill in skill_names:
            for prediction in predictions:
                text = '========================\n'
                text += '\n'.join(f'**{k}**: {v}' for k, v in prediction.items() if k not in skill_names + ['index'])
                text += f'\n\n__**{skill}**__: {prediction[skill]}'
                ground_truth = GroundTruth(prediction_id=prediction['index'], skill_name=skill)

                message = await channel.send(
                    text, view=AcceptRejectView(
                        prediction_id=ground_truth.prediction_id,
                        skill_name=ground_truth.skill_name)
                )
                ground_truths.append(ground_truth)
                await db.execute('''
                    INSERT INTO discord_gt_message (prediction_id, skill_name, message_id)
                    VALUES (?, ?, ?)
                ''', (ground_truth.prediction_id, ground_truth.skill_name, message.id))
                await db.commit()

        # TODO: do we need to store it in advance?
        await self.store_ground_truths(ground_truths, db)


app = DiscordAPI()


@app.on_event("startup")
async def on_startup():
    await app.init_db()
    await app.init_db_gt_message()
    app.start_discord_bot()


@app.on_event("shutdown")
def on_shutdown():
    app.stop_discord_bot()