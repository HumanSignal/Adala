import React from 'react';
import { AdalaAPI, SubmitRequest} from "./_api";

interface AdalaSubmitInterface {
  inputFile: string;
  outputFile: string;
  errorFile: string;
  instructions: string;
  labels: string[];
  model: string;
  apiKey: string;
}

interface AdalaGetStatusInterface {
  jobId: string;
}

interface AdalaCancelInterface {
  jobId: string;
}


export class Adala {
  private apiClientInstance: any;

  constructor(url: string) {
    this.apiClientInstance = new AdalaAPI({BASE: url});
  }

  // submit should accept an input of type Submit
  async submit(req: AdalaSubmitInterface): Promise<any> {
    try {
      const response = await this.apiClientInstance.default.submitSubmitPost({
        requestBody: {
          agent: {
            environment: {
              type: "FileStreamAsyncKafkaEnvironment",
              kafka_bootstrap_servers: "kafka:9093",
              // kafka_bootstrap_servers: "localhost:9093",
              kafka_input_topic: "adala-input",
              kafka_output_topic: "adala-output",
              input_file: req.inputFile,
              output_file: req.outputFile,
              error_file: req.errorFile,
              pass_through_columns: null
            },
            skills: [{
              // type: "ClassificationSkill",
              type: "TransformSkill",
              name: "text_classifier",
              // In the first version, we don't use the instructions (all prompts go to the input_template). Consider using it for the efficient prefill phase in the future.
              instructions: "",
              input_template: req.instructions,
              output_template: "{output}",
              labels: {
                output: req.labels
              }
            }],
            runtimes: {
              default: {
                type: "AsyncOpenAIChatRuntime",
                model: req.model,
                api_key: req.apiKey,
                max_tokens: 10,
                temperature: 0,
                concurrent_clients: 100,
                batch_size: 100,
                timeout: 10,
                verbose: false
              }
            }
          },
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error submitting request:', error);
      throw error;
    }
  }

  // Example method for getting the status
  async getStatus(req: AdalaGetStatusInterface): Promise<any> {
    try {
      const response = await this.apiClientInstance.default.getStatusGetStatusPost({
        requestBody: {
          job_id: req.jobId
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error getting status:', error);
      throw error;
    }
  }

  async cancel(req: AdalaCancelInterface): Promise<any> {
    try {
      const response = await this.apiClientInstance.default.cancelJobCancelPost({
        requestBody: {
          job_id: req.jobId
        }
      });
      return response.data;
    } catch (error) {
      console.error('Error getting status:', error);
      throw error;
    }
  }
}

export default Adala;
