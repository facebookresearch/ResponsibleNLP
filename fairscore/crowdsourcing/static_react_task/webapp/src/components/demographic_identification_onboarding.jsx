/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React, { useState } from "react";
import { DropdownButton, Dropdown, Form } from "react-bootstrap";
import { Directions } from "./utils.jsx"
import { DemographicSurvey } from "./demographic_survey.jsx"
import ToggleButton from 'react-bootstrap/ToggleButton'
import ToggleButtonGroup from 'react-bootstrap/ToggleButtonGroup'

function RenderStringOrButton(word) {
  var regExp = /[a-zA-Z]/g;
  if (regExp.test(word) == false) {
    return (
      <text className="button-no-click button subtitle selectWords">{word}</text>
    );
  } else {
    return (
    <ToggleButton className="button subtitle selectWords" value={word}>
      {word}
    </ToggleButton>
    );
  }
}

function DemographicAnnotation({text, axis, spans, setSpans}) {
    return (
        <div className="annotateDemographic">
          <div className="field">
            <div className="field is-grouped subtitle is-5">
              <b> Are there words identifying <text className="red">{axis}</text> in this sentence?</b>
              <form>
                  <div className="radio">
                <label>
                <input type="radio" value="Yes" name={axis} onClick={() => console.log("yes")} /> Yes
                </label>
                </div>
                <div className="radio">
                <label> 
                <input type="radio" value="No" name={axis} onClick={() => console.log("no")} /> No
                </label>
                </div>
              </form>
            </div>
          </div>
          <div className="field">
          <p className="subtitle is-5">Select <b>every</b> word that corresponds to the demographic group <b className="red">{axis}</b>.</p>
          <ToggleButtonGroupControlled sentence={text} spans={spans} setSpans={setSpans} />

          </div>
        </div>
    )
}

function ToggleButtonGroupControlled({sentence, spans, setSpans}) {
    const wordsList = sentence.split(" ")
  
    const handleChange = (val, event) => {
        setSpans(val);
    }
  
    return (
      <ToggleButtonGroup className="checkbox mb-2" type="checkbox" value={spans} onChange={handleChange} >
        {wordsList.map((word) => RenderStringOrButton(word))}
      </ToggleButtonGroup>
    );
  }

function DemographicOnboardingComponent({ onSubmit }) {
    const text = "Please write in clear English sentences that are free of grammatical errors and have full punctuation.\n \
    Avoid racism, sexism, and other discriminatory or offensive language. \n \
    Do not provide any personal information in your sentences, such as your name, address, email or phone number. \n \
    Incomplete work will not be rewarded. \n \
    You must click “submit” when your work is complete.";
    const  [state,setState]=useState("Choose a demographic axis");
    const response = { "spans": {}, "survey": {"gender": [], "race": []}}

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
  
    function handleUpdate(demographic) {
        setState(demographic);
    }
  
    function handleSubmit() {
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

      // Set selected spans
      response["spans"]["gender"] = genderSpans
      response["spans"]["race"] = raceSpans
      response["spans"]["age"] = ageSpans
      console.log(response)
      onSubmit(response)
    }

    const sentence1 = "Jessica cooks dinner for her young friends every night in his house ."
    const sentence2 = "Your grandfather is a famous asian activist today , but he almost gave up several times ."

    return (
      <div>
        <Directions>
        <p className="title is-3 is-spaced">Select Words</p>
        <div className="instructions">
          <div>The goal of this task is to select words that contain demographic information. You will be shown a demographic axis and will click words that are associated with that axis.</div>
            <p><b>Here are some examples:</b></p>
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
          <div>
  
            <p>After completing your task, you will be asked to fill out an optional short demographic survey. Your responses are anonymous and will only be used in aggregate. Note that whether or not you choose to fill out the survey will not affect payment towards the HIT. </p>
          </div>
          <div className="reminders">{text}</div>
          </div>
          <div>
            <b>IMPORTANT:</b> Your annotations will be used by the requestor and others for research purposes. This includes public disclosure of the work as part of research data sets and research papers. Please ensure that your work does not contain personal information, such as, for example, your name, address, email, or phone number, or other information that you would wish to keep private.
  </div>
        </Directions>
        <section className="section">
        <div className="container">
          <p className="subtitle is-5">For each sentence shown, answer the following questions to identify words that reveal demographic information in this sentence, or select <b>"No"</b> if there aren't any.</p>
          <p className="subtitle is-3 is-spaced">{sentence1}</p>
          <DemographicAnnotation text={sentence1} axis={"gender"} spans={genderSpans} setSpans={setGenderSpans} />
          <DemographicAnnotation text={sentence1} axis={"race/ethnicity"} spans={raceSpans} setSpans={setRaceSpans} />
          <DemographicAnnotation text={sentence1} axis={"age"} spans={ageSpans} setSpans={setAgeSpans} />
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

export { DemographicOnboardingComponent as DemographicOnboardingComponent };