/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React, { useState } from "react";
import { DropdownButton, Dropdown, Form, ButtonGroup } from "react-bootstrap";
import { OnboardingComponent } from "./rewrite_sentence_onboarding.jsx"


function LoadingScreen() {
  return <Directions>Loading...</Directions>;
}

function DemographicSurvey({
  genderFields, 
  setGenderFields, 
  raceFields, 
  setRaceFields,
  textFields,
  setTextFields,
}) {
  function handleDemographicChange(event) {
    const value = event.target.value
    const formName = event.currentTarget.id
    const field = event.currentTarget.name
    const isChecked = event.currentTarget.checked
    if (formName == "gender") {
      setGenderFields({...genderFields, [field] : isChecked });
    } else if (formName == "race") {
      setRaceFields({...raceFields, [field] : isChecked });
    } else if (
      formName == "self-described" ||
      formName == "age"
    ) {
      setTextFields({...textFields, [formName]: value})
    }
  }

  return (
  <section className="hero is-light">
  <Form className="survey">
    <h2>
      Demographic Survey (Optional)
    </h2>
    <Form.Label className="label">Please state your gender (check all that apply)</Form.Label>
    <div key={`gender-checkbox`} className="mb-3">
      <Form.Check
        type="checkbox"
        id={`gender`}
        label={`Woman`}
        name="female"
        onChange={e => handleDemographicChange(e)}
      />  
      <Form.Check 
        type="checkbox"
        id={`gender`}
        label={`Man`}
        name="male"
        onChange={e => handleDemographicChange(e)}
      />
      <Form.Check 
        type="checkbox"
        id={`gender`}
        label={`Non-binary`}
        name="non-binary"
        onChange={e => handleDemographicChange(e)}
      />
      <Form.Check 
        type="checkbox"
        id={`gender`}
        label={`Self described`}
        name="other"
        onChange={e => handleDemographicChange(e)}
      />
      <Form.Check 
        type="checkbox"
        id={`gender`}
        label={`Prefer not to answer`}
        name="na"
        onChange={e => handleDemographicChange(e)}
      />
    </div>
    <Form.Label className="label">If gender is self described, please enter below:
    </Form.Label>
    <Form.Control 
      type="text" 
      placeholder="Describe gender" 
      id={`self-described`}
      onChange={e => handleDemographicChange(e)}
    />

    <Form.Label className="label">Please state your race (check all that apply):
</Form.Label>
    <div key={`race-checkbox`} className="mb-3"> 
      <Form.Check
        type="checkbox"
        id={`race`}
        label={`Hispanic or Latino`}
        name="hispanic"
        onChange={e => handleDemographicChange(e)}
      /> 
      <Form.Check 
        type="checkbox"
        id={`race`}
        label={`Black or African American`}
        name="black"
        onChange={e => handleDemographicChange(e)}
      />
      <Form.Check
        type="checkbox"
        id={`race`}
        label={`White`}
        name="white"
        onChange={e => handleDemographicChange(e)}
      /> 
      <Form.Check 
        type="checkbox"
        id={`race`}
        label={`American Indian or Alaska Native`}
        name="native-american"
        onChange={e => handleDemographicChange(e)}
      />
      <Form.Check 
        type="checkbox"
        id={`race`}
        label={`Asian`}
        name="asian"
        onChange={e => handleDemographicChange(e)}
      />
      <Form.Check 
        type="checkbox"
        id={`race`}
        label={`Native Hawaiian or Other Pacific Islander`}
        name="pacific"
        onChange={e => handleDemographicChange(e)}
      />
      <Form.Check 
        type="checkbox"
        id={`race`}
        label={`Prefer not to answer`}
        name="na"
        onChange={e => handleDemographicChange(e)}
      />
    </div>
    <Form.Label className="label">Please enter your age in years:
</Form.Label>
<Form.Control 
  id={`age`}
  type="number" 
  placeholder="Age"
  onChange={e => handleDemographicChange(e)}
/>
</Form>
</section>);
}

function Feedback({setFeedback}) {
  return (
    <div className="section">
      <div className="field">
        <p className="font-size-med">If you have any feedback for us, or if there was a mistake in your task, please tell us here:</p>
      </div>
      <textarea 
      className="font-size-small"
      type="text" 
      rows="3"
      style={{ minWidth: '60%' }} 
      onChange={e => setFeedback(e.target.value)} 
      />
    </div>
  )
}

function Directions({ children }) {
  return (
    <section className="hero is-light">
      <div className="hero-body">
        <div className="container">
          <p className="subtitle is-5">{children}</p>
        </div>
      </div>
    </section>
  );
}


function SimpleFrontend({ taskData, isOnboarding, onSubmit, onError }) {
  if (!taskData) {
    return <LoadingScreen />;
  }
  if (isOnboarding) {
    return <OnboardingComponent onSubmit={onSubmit} />;
  }
  // Starting a timer to track elapsed time
  const start = window.performance.now()
  // State variables
  const [showMoreText, setShowMoreText] = React.useState(false)
  const  [perturbed,setPerturbedState]=useState("");

  // Collecting demographic survey responses
  const [genderFields, setGenderFields] = useState({});
  const [raceFields, setRaceFields] = useState({});
  const [textFields, setTextFields] = useState({});

  // Free form feedback, optional
  const  [feedback,setFeedback]=useState("");

  const text = "Please write in clear English sentences that are free of grammatical errors and contain full punctuation.\n \
  Avoid racism, sexism, or other discriminatory or offensive language. \n \
  Do not provide any personal information in your sentences, such as your name, address, email or phone number. \n \
  Incomplete work will not be rewarded. \n \
  You must click “submit” when your work is complete.";
  const response = { "rewrite": "", "survey": {"gender": [], "race": []}, "feedback": ""}

  function RenderSnippet(tokens, x_pos, y_pos) {
    tokens[x_pos][y_pos] = "<text id='highlight'>" + tokens[x_pos][y_pos] + "</text>"
    return tokens.map(x => x.join(" ")).join("\n")
  }

  function handleSubmit() {
    // If string is all whitespace, ask to complete the task
    if (!/\S/.test(perturbed)) {
      alert("Please complete the task before submitting.")
      return
    }
    // If string is very short, ask to complete the task
    const originalLength = taskData.text.split(" ").length
    const rewriteLength = perturbed.split(" ").length
    if (rewriteLength < (originalLength * 0.8)) {
      alert("Please rewrite the full text before submitting.")
      return
    }
    // If perturbed category is age related, and the resulting string contains the exact same word, send a warning to the worker.
    if (/child \(< 18\)|young \(18-44\)|middle-aged \(45-64\)|senior \(65\+\)|adult \(unspecified\)/.test(perturbed.toLowerCase())) {
      alert("Please re-read the task and try again. Ensure that your rewrite is logically and grammatically correct.")
      return
    }
    // If the worker has spent less than 90 seconds, ask to proofread.
    const end = window.performance.now()
    const timeElapsed = (end - start) / 1000
    let threshold = 3;
    if (timeElapsed < threshold) {
      alert("Before you submit, please take some time to read through the instructions and text again carefully. Ensure that your rewrite is logically and grammatically correct.")
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
    response["rewrite"] = perturbed
    response["feedback"] = feedback
    console.log(response)
    onSubmit(response)
  }
  const showExpanded = true
  let additionalInstructions = (
    <div>
      <div><p><b>Here are some examples of swaps:</b></p></div>
      <div className="examples">
      “<text id="highlight">Alice</text> is cooking dinner for her children.”<br />
      Demographic axis: <text className="red">gender</text> <br />
      Instruction: <i>Please change the word <text id="highlight">Alice</text> and all references to <text id="highlight">Alice</text> from <text className="red">female</text> to <text className="red">male</text>. </i><br />
      Rewrite: “<text id="highlight">Adam</text> is cooking dinner for <text id="highlight">his</text> children.” <br />
      </div>
      <div className="examples">
        <p>“The <text id="highlight">girl</text> likes to play with her dolls all day.”</p>
        <p>Demographic axis: <text className="red">gender</text></p>
        Instruction: <i>Please change the word <text id="highlight">girl</text> and all references to <text id="highlight">girl</text> from <text className="red">female</text> to <text className="red">non-binary</text>. </i><br />
        <p>Rewrite: “The <text id="highlight">child</text> likes to play with <text id="highlight">their</text> dolls all day.”</p>
      </div>
      <div className="examples">
        <p>“The <text id="highlight">Asian</text> engineer quickly codes up an algorithm.”</p>
        <p>Demographic axis: <text className="red">ethnicity</text></p>
        Instruction: <i>Please change the word <text id="highlight">Asian</text> and all references to <text id="highlight">Asian</text> from <text className="red">Asian</text> to <text className="red">Black or African American</text>. </i><br />
        <p>Rewrite: “The <text id="highlight">Black</text> engineer quickly codes up an algorithm.”</p>
      </div>
      <div className="instructions">
        The rewritten sentence should only change references to the person identified, so this is <b>NOT</b> a valid swap:
        <div className="examples">
        <p>“The <text id="highlight">husband</text> comes home to his wife every night.”</p>
        Instruction: <i>Please change the word <text id="highlight">husband</text> and all references to <text id="highlight">husband</text> from <text className="red">male</text> to <text className="red">female</text>. </i><br />
        <p>Demographic axis: <text className="red">gender</text></p>
        <p>Incorrect rewrite: “The <text id="highlight">wife</text> comes home to <text id="highlight">her</text> <text id="highlight">husband</text> every night.” &#10060;</p>
        <p>Correct rewrite: “The <text id="highlight">wife</text> comes home to <text id="highlight">her</text> wife every night." &#9989; </p>
      </div>
      </div>
      <div>
        <p>When considering words to swap, try not to be influenced by your default assumptions! </p><br />

        <p>After completing your task, you will be asked to fill out an optional short demographic survey. Your responses are anonymous and will only be used in aggregate. Note that whether or not you choose to fill out the survey will not affect payment towards the HIT.</p>
      </div>
      <button
        className="button"
        onClick={() => setShowMoreText(false)}
      >
        Show less
              </button>
    </div>
  );

  console.log(taskData)

  return (
    <div>
      <Directions>
        <p className="title is-3 is-spaced">Rewrite sentences</p>
        <div className="instructions">
          <div>In this task, you will be shown a sentence, and given a word in the sentence that contains demographic information about a person(s), such as ethnicity, gender or age.</div>
          <div>You will then change any words in the sentence that identify that person(s) to refer to a different demographic group, while ensuring fluency and grammatical correctness.</div>
          <div>If the highlighted word does not refer to a person, please write the sentence as is.</div>
        </div>

        {showMoreText ? additionalInstructions : <button
          className="button"
          onClick={() => setShowMoreText(true)}
        >
          Show more
              </button>}

        <div className="reminders red">{text}</div>
        <div>
          <b>IMPORTANT:</b> Your annotations will be used by the requestor and others for research purposes. This includes public disclosure of the work as part of research data sets and research papers. Please ensure that your work does not contain personal information, such as, for example, your name, address, email, or phone number, or other information that you would wish to keep private.
</div>
      </Directions>
      <section className="section">
        <div className="container">
          <p className="subtitle is-5">Rewrite this sentence to change the demographic group for the person indicated by the highlighted word, while keeping the meaning as close to the original as possible (while ensuring fluency and grammatical correctness).</p>
          <p className="subtitle is-3 is-spaced"><div className="display-linebreak" dangerouslySetInnerHTML={{ __html: RenderSnippet(taskData.text_newlines, taskData.x_pos, taskData.y_pos) }}></div></p>
          <div className="field">
            <div className="field">
              <p className="font-size-med"><b>Demographic Axis: <text className="red">{taskData["demographic_axis"]}</text></b></p>
            </div>
            <div className="field">
              <p className="font-size-med"><b>Please change <text id="highlight">{taskData["selected_word"]}</text> and all references to <text id="highlight">{taskData.selected_word}</text> from <text className="red">{taskData.selected_word_category}</text> to <text className="red">{taskData["perturbed_word_category"]}</text>, including names where appropriate.</b> Please keep SEP tags and other markers if they appear in the original text. If the highlighted word does not refer to a person, please write the sentence as is.</p>
            </div>
            <div>
              <p className="font-size-med">Rewritten sentence:</p>
              <textarea 
              className="font-size-large"
              type="text" 
              rows="24"
              style={{ minWidth: '90%' }} 
              id={`rewrite`}
              name="rewrite"
              onChange={e => setPerturbedState(e.target.value)} 
              // onPaste={e => e.preventDefault()}
              />
            </div>
          </div>
        <Feedback
          setFeedback={setFeedback}
        >
        </Feedback>
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

export { LoadingScreen, SimpleFrontend as BaseFrontend };
