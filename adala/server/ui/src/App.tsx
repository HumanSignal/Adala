import React, {useState, useEffect} from "react";
import "./App.css";
import {Adala} from "./adala";
import testData from "./testData";

// const testData = {
//   adala_server_url: "http://localhost:30001",
//   inputFile: "s3://hs-sandbox-pub/sales_5samples.csv",
//   outputFile: "s3://hs-sandbox-pub/sales_5samples_output.csv",
//   errorFile: "s3://hs-sandbox-pub/sales_5samples_error.csv",
//   instructions: `
//   Given a brief account of a lost sales deal, classify the reason for loss into one of the following categories:
//    Feature Lack, Price, Integration Issues, Usability Concerns, or Competitor Advantage.
//   `,
//   labels: ["Feature Lack", "Price", "Integration Issues", "Usability Concerns", "Competitor Advantage"],
//   model: "gpt-3.5-turbo-0125",
//   apiKey: "sk-...",
// }

type FormData = {
  inputFile: string;
  outputFile: string;
  errorFile: string;
  instructions: string;
  labels: string[]; // Explicitly typing labels as an array of strings
  model: string;
  apiKey: string;
};

const App = () => {
  const [formData, setFormData] = useState<FormData>(testData);
  const [jobId, setJobId] = useState("");
  const [token, setToken] = useState(""); // Add token state to store token for prediction stream
  const [predictions, setPredictions] = useState(""); // Add predictions state to store prediction stream records
  const [progress, setProgress] = useState(0);

  const adala = new Adala(testData.adala_server_url);
  const topic = "adala-output"; // Add topic for prediction stream
  const totalPredictions = 1000;

  console.log('Token:', token)
  console.log('JobId:', jobId)
  const handleSubmit = async () => {
    try {
      const response = await adala.submit({
        ...formData,
        labels: formData.labels,
      });
      setJobId(response.job_id);
      setToken(response.token); // Set token state with token from response
      // checkStatusPeriodically(response.job_id);
    } catch (error) {
      console.error("Error submitting data to server:", error);
    }
  };

  // Function to read prediction stream records from Kafka
  const processPredictions = (data: any) => {
    setPredictions(prevState => {
      // Assuming each call to this function represents one "chunk" of data being received
      const updatedPredictions = prevState + data + "\n\n";

      // Update progress based on the number of "\n\n" occurrences, assuming each chunk ends with "\n\n"
      const chunksReceived = updatedPredictions.split("\n\n").length - 1;
      const newProgress = Math.min((chunksReceived / totalPredictions) * 100, 100); // Ensure progress does not exceed 100%

      setProgress(newProgress);
      return updatedPredictions;
    });
  }

  useEffect(() => {
    if (jobId && token) {
      adala.getPredictionStream(
        {jobId, token, topic},
        processPredictions
      )
    }
  }, [jobId, token]);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const {name, value} = event.target;
    setFormData(prevState => ({
      ...prevState,
      // for list based fields (labels, passThroughColumns), split the value by comma and trim each item
      [name]: name === "labels" ? value.split(",").map(item => item.trim()) : value,
    }));
  };

  return (
    <div className="app-container">
      <div className="left-panel">
        {/* Text inputs for each field */}
        {Object.entries(formData).map(([key, value]) => {
          if (key !== "labels") {
            return (
              <input
                key={key}
                type="text"
                name={key}
                value={value}
                onChange={handleChange}
                placeholder={key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1').trim()} // Formatting key to human-readable form
              />
            );
          } else {
            return (
              <input
                key={key}
                type="text"
                name={key}
                value={Array.isArray(value) ? value.join(", ") : ""}
                onChange={handleChange}
                placeholder="Comma-separated list of items"
              />
            );
          }
        })}
        <button onClick={handleSubmit}>Submit</button>
      </div>
      <div className="right-panel">
        <div className="progress-bar-container" style={{width: '100%', backgroundColor: '#ddd'}}>
          <div className="progress-bar" style={{width: `${progress}%`, backgroundColor: 'blue', height: '20px'}}></div>
        </div>

        {/*/!*Display job id and status*!/*/}
        <p>Job ID: {jobId}</p>
        <textarea readOnly value={predictions} placeholder="Status will be displayed here"/>

      </div>
    </div>
  );
};

export default App;
