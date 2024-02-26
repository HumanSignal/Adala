import React, { useState } from "react";
import "./App.css";
import { Adala } from "./adala";
// import testData from "./testData";

const testData = {
  inputFile: "s3://hs-sandbox-pub/sales_5samples.csv",
  outputFile: "s3://hs-sandbox-pub/sales_5samples_output.csv",
  errorFile: "s3://hs-sandbox-pub/sales_5samples_error.csv",
  instructions: `
  Given a brief account of a lost sales deal, classify the reason for loss into one of the following categories:
   Feature Lack, Price, Integration Issues, Usability Concerns, or Competitor Advantage. 
  `,
  labels: ["Feature Lack", "Price", "Integration Issues", "Usability Concerns", "Competitor Advantage"],
  model: "gpt-3.5-turbo-0125",
  apiKey: "sk-...",
}

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
  const [status, setStatus] = useState("");
  const [jobId, setJobId] = useState("");
  const adala = new Adala("http://localhost:30001");

  const handleSubmit = async () => {
    try {
      const response = await adala.submit({
        ...formData,
        labels: formData.labels,
      });
      setJobId(response.job_id);
      checkStatusPeriodically(response.job_id);
    } catch (error) {
      console.error("Error submitting data to server:", error);
    }
  };

  const checkStatusPeriodically = (currentJobId: string) => {
    const intervalId = setInterval(async () => {
      try {
        const response = await adala.getStatus({ jobId: currentJobId });
        setStatus(response.status); // Adjust this according to your actual response structure
        if (response.status === "SUCCESS" || response.status === "FAILURE") {
          clearInterval(intervalId);
        }
      } catch (error) {
        console.error("Error fetching status from server:", error);
        clearInterval(intervalId);
      }
    }, 5000); // Poll every 5 seconds
  };

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
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
          if (key !== "labels")
          {
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
        {/*Display job id and status*/}
        <p>Job ID: {jobId}</p>
        <textarea readOnly value={status} placeholder="Status will be displayed here" />
      </div>
    </div>
  );
};

export default App;
