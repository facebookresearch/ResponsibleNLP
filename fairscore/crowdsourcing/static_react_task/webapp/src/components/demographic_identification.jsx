/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React, { useState } from "react";
import ToggleButton from 'react-bootstrap/ToggleButton'
import ToggleButtonGroup from 'react-bootstrap/ToggleButtonGroup'
import { DropdownButton, Dropdown, Form, ButtonGroup } from "react-bootstrap";
import { DemographicOnboardingComponent } from "./demographic_identification_onboarding.jsx"
import { DemographicSurvey } from "./demographic_survey.jsx"
import { Directions } from "./utils.jsx"


function LoadingScreen() {
  return <Directions>Loading...</Directions>;
}

function RenderStringOrButton(word, idx) {
  var regExp = /[a-zA-Z0-9]/g;
  if (regExp.test(word) == false) {
    return (
      <text className="button-no-click button subtitle selectWords">{word}</text>
    );
  } else {
    return (
    <ToggleButton className="button subtitle selectWords" value={idx}>
      {word}
    </ToggleButton>
    );
  }
}

function ToggleButtonGroupControlled({tokens, spans, setSpans}) {
    const wordsList = tokens
    const handleChange = (val, event) => {
        console.log("setting the span")
        setSpans(val);
    }
  
    return (
      <ToggleButtonGroup className="checkbox mb-2" type="checkbox" value={spans} onChange={handleChange} >
        {wordsList.map((word, idx) => RenderStringOrButton(word, idx))}
      </ToggleButtonGroup>
    );
  }

function DemographicAnnotation({tokens, axis, spans, setSpans, setAxis}) {
    return (
        <div className="annotateDemographic">
          <div className="field">
            <div className="field is-grouped subtitle is-5">
              <b> Are there words identifying <text className="red">{axis}</text> in this sentence?</b>
              <form>
                  <div className="radio">
                <label>
                <input type="radio" value="Yes" name={axis} onClick={() => setAxis(true)} /> Yes
                </label>
                </div>
                <div className="radio">
                <label> 
                <input type="radio" value="No" name={axis} onClick={() => setAxis(false)} /> No
                </label>
                </div>
              </form>
            </div>
          </div>
          <div className="field">
          <p className="subtitle is-5">Select <b>every</b> word that corresponds to the demographic group <b className="red">{axis}</b>.</p>
          <ToggleButtonGroupControlled tokens={tokens} spans={spans} setSpans={setSpans} />

          </div>
        </div>
    )
}

function DemographicTask({ taskData, isOnboarding, onSubmit, onError }) {
  if (!taskData) {
    return <LoadingScreen />;
  }
  if (isOnboarding) {
    return <DemographicOnboardingComponent onSubmit={onSubmit} />;
  }
  // Starting a timer to track elapsed time
  const start = window.performance.now()
  // State variables
  const [showMoreText, setShowMoreText] = React.useState(false)
  // Collecting demographic survey responses
  const [genderFields, setGenderFields] = useState({});
  const [raceFields, setRaceFields] = useState({});
  /** NOTE: textFields captures free form text responses,
   * Including age and self described gender. */
  const [textFields, setTextFields] = useState({});

  // Storing span selection responses
  const [genderSpans, setGenderSpans] = useState([]);
  const [raceSpans, setRaceSpans] = useState([]);
  const [ageSpans, setAgeSpans] = useState([]);

  // Demographic axis state
  const [genderAxis, setGenderAxis] = useState(null);
  const [raceAxis, setRaceAxis] = useState(null);
  const [ageAxis, setAgeAxis] = useState(null);

  const text = "Please write in clear English sentences that are free of grammatical errors.\n \
  Avoid racism, sexism, or other discriminatory or offensive language. \n \
  Do not provide any personal information in your sentences, such as your name, address, email or phone number. \n \
  Incomplete work will not be rewarded. \n \
  You must click “submit” when your work is complete.";
  const response = { "spans": {}, "demographic_axis": {}, "survey": {"gender": [], "race": []}}

  function handleSubmit() {
    const end = window.performance.now()
    const timeElapsed = (end - start) / 1000
    let threshold = 5
    if (timeElapsed < threshold) {
      alert("Before you submit, please take some time to double check that the words you have highlighted for each axis are correct.")
      return
    }
    // parse checked survey responses
    for (const [key, value] of Object.entries(genderFields)) {
      if (value == true) {
        response["survey"]["gender"].push(key)
      }
    }
    for (const [key, value] of Object.entries(raceFields)) {
      if (value == true) {
        response["survey"]["race"].push(key)
      }
    }
    for (const [key, value] of Object.entries(textFields)) {
      if ((value !== "") && (value !== null)) {
        response["survey"][key] = value
      }
    }
    // If there are no responses, ask to redo the task
    if (genderAxis == null || raceAxis == null || ageAxis == null) {
      alert("Please complete all parts of the task before submitting.")
      return
    }
    // If there are fewer than 2% selected words, ask to redo the task
    const numTotalWords = taskData.tokens.length
    const numSelectedWords = genderSpans.length + raceSpans.length + ageSpans.length
    if ((numSelectedWords / numTotalWords) < 0.02) {
      alert("Please ensure that you have selected all demographic identifying terms.")
      return
    }
    // Collect Axis Annotations
    response["demographic_axis"]["gender"] = genderAxis
    response["demographic_axis"]["race"] = raceAxis
    response["demographic_axis"]["age"] = ageAxis 
    
    // Set selected spans
    response["spans"]["gender"] = genderSpans.map((index) => [taskData.tokens[index], index])
    response["spans"]["race"] = raceSpans.map((index) => [taskData.tokens[index], index])
    response["spans"]["age"] = ageSpans.map((index) => [taskData.tokens[index], index])

    console.log(response)
    onSubmit(response)
  }
  const showExpanded = true
  let additionalInstructions = (
    <div>
      <div><p><b>Here are some examples:</b></p></div>
      <div className="examples">
  Demographic axis: <b>Age</b> <br />
  Text: “The <b className="red">old</b> man is cooking dinner for his <b className="red">grandchildren</b>.”<br />
          </div>
          <div className="examples">
            <p>Demographic axis: <b>Gender</b></p>
            <p>Text: “<b className="red">John</b> says their <b className="red">daughter</b> likes to play with <b className="red">her</b> toys all day.”</p>
          </div>
          <div className="examples">
            <p>Demographic axis: <b>Race/Ethnicity</b></p>
            <p>Text: “She is part of a group teaching <b className="red">Black</b> girls to code.”</p>
          </div>
      <button
        className="button"
        onClick={() => setShowMoreText(false)}
      >
        Show less
              </button>
    </div>
  );

  return (
    <div>
      <Directions>
        <p className="title is-3 is-spaced">Select Words</p>
        <div className="instructions">
          <div>The goal of this task is to select words that contain demographic information about people. You will be shown a demographic axis and will click words that are associated with that axis.</div>
          <div>If there are no words for the given demographic axis, please check "no". </div>
        </div>

        {showMoreText ? additionalInstructions : <button
          className="button"
          onClick={() => setShowMoreText(true)}
        >
          Show more
              </button>}
        <div>
          <p>After completing your task, you will be asked to fill out an optional short demographic survey. Your responses are anonymous and will only be used in aggregate. Note that whether or not you choose to fill out the survey will not affect payment towards the HIT.</p>
        </div>
        <div className="reminders red">{text}</div>
        <div>
          <b>IMPORTANT:</b> Your annotations will be used by the requestor and others for research purposes. This includes public disclosure of the work as part of research data sets and research papers. Please ensure that your work does not contain personal information, such as, for example, your name, address, email, or phone number, or other information that you would wish to keep private.
</div>
      </Directions>
      <section className="section">
        <div className="container">
          <p className="subtitle is-5">For the sentence shown, answer the following questions to identify words that reveal demographic information about people, or select <b>"No"</b> if there aren't any.</p>
          <p className="subtitle is-3 is-spaced">{taskData.text}</p>
          <DemographicAnnotation tokens={taskData.tokens} axis={"gender"} spans={genderSpans} setSpans={setGenderSpans} setAxis={setGenderAxis} />
          <DemographicAnnotation tokens={taskData.tokens} axis={"race/ethnicity"} spans={raceSpans} setSpans={setRaceSpans} setAxis={setRaceAxis} />
          <DemographicAnnotation tokens={taskData.tokens} axis={"age"} spans={ageSpans} setSpans={setAgeSpans} setAxis={setAgeAxis} />
          <div className="control">
            {/* TODO: this is hard coded */}
            <button
              className="button is-link"
              onClick={handleSubmit}
            >
              Submit
              </button>
          </div>
        </div>
      </section>
    </div>
  );
}

export { LoadingScreen, DemographicTask as BaseFrontend };