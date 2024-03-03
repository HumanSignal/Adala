import React from 'react';
import {AdalaAPI, SubmitRequest} from "./_api";

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

interface AdalaGetPredictionStreamInterface {
  jobId: string;
  token: string;
  topic: string;
}

export class Adala {
  private url: string;
  private apiClientInstance: any;

  constructor(url: string) {
    this.url = url;
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
              // kafka_bootstrap_servers: "kafka:9093",
              kafka_bootstrap_servers: "localhost:9093",
              kafka_input_topic: "adala-input",
              kafka_output_topic: "adala-output",
              input_file: req.inputFile,
              output_file: req.outputFile,
              error_file: req.errorFile,
              pass_through_columns: null
            },
            skills: [{
              type: "ClassificationSkill",
              // type: "TransformSkill",
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

  // method for reading data from /prediction-steam endpoint using EventSource and SSE
  // query args: job_id, token, topic
  async getPredictionStream(
    req: AdalaGetPredictionStreamInterface,
    onReceive: (data: any) => void,
  ): Promise<any> {
    const url = `${this.url}/prediction-stream?job_id=${encodeURIComponent(req.jobId)}&token=${req.token}&topic=${req.topic}`;
    const eventSource = new EventSource(url);
    console.log("Prediction stream URL:", url);

    eventSource.onopen = (e) => {
      console.log('Connection is established: ', e);
    }

    eventSource.onerror = (e) => {
      // console.log error message
      console.error('EventSource failed:', e);
      eventSource.close();
    }

    eventSource.onmessage = (e: MessageEvent) => {
      console.log("Received data from prediction stream:", e.data);
      onReceive(e.data);
    }

    return eventSource;
  }

}

export default Adala;
