// @ts-nocheck

import React, {useState, useEffect} from "react";
import "./App.css";
import { Adala } from "./adala";

const App = () => {
    const [input, setInput] = useState("");
    const [status, setStatus] = useState("");
    const [jobId, setJobId] = useState(""); // This should be set to the jobId returned from the submit call
    const adala = new Adala("http://localhost:30001");

    const handleSubmit = async () => {
      try {
        const response = await adala.submit({
          inputFile: "input.txt",
          outputFile: "output.txt",
          errorFile: "error.txt",
          instructions: input,
          labels: ["positive", "negative"],
          model: "gpt-3.5-turbo",
          apiKey: "your-api-key"
        });
        setJobId(response.data.jobId);
        checkStatusPeriodically();
      } catch
        (error) {
        console.error("Error submitting data to server:", error);
      }
    };

    const checkStatusPeriodically = () => {
      const intervalId = setInterval(async () => {
        try {
          const response = await adala.getStatus({jobId});
          setStatus(response.data.status);
          if (response.data.status === "Complete" || response.data.status === "Error") {
            clearInterval(intervalId);
          }
        } catch (error) {
          console.error("Error fetching status from server:", error);
          clearInterval(intervalId);
        }
      }, 5000); // Poll every 5 seconds
    };

    return (
      <div className="app-container">
        <div className="left-panel">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Enter your input"
          />
          <button onClick={handleSubmit}>Submit</button>
        </div>
        <div className="right-panel">
          <textarea readOnly value={status} placeholder="Status will be displayed here"/>
        </div>
      </div>
    );
  }
;

export default App;
