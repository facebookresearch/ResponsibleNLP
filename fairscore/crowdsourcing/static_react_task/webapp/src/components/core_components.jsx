/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React, { useState } from "react";
import { DropdownButton, Dropdown, Form } from "react-bootstrap";


function OnboardingComponent({ onSubmit }) {
  const text = "Please write in clear English sentences that are free of grammatical errors and have full punctuation.\n \
  Avoid racism, sexism, or other discriminatory or offensive language. \n \
  Do not provide any personal information in your sentences, such as your name, address, email or phone number. \n \
  Incomplete work will not be rewarded. \n \
  You must click “submit” when your work is complete.";
  const  [state,setState]=useState("Choose a demographic axis");
  const response = { "demographic": "", "rewrite": "", "questions": { "ethnicity_perturb": "", "gender_perturb": "" } }

  function handleUpdate(demographic) {
      setState(demographic);
  }

  function handleSubmit() {
    console.log(response)
    response["demographic"] = state
    onSubmit(response)
  }
  return (
    <div>
      <Directions>
        <p className="title is-3 is-spaced">Rewrite sentences</p>
        <div className="instructions">The goal of this taskk is to rewrite sentences by swapping words or phrases along certain demographic axis while preserving the meaning of the sentence.
      <br />
          <div>You will be shown a sentence, and asked to identify whether the sentence contains references to any demographic group, such as ethnicity or gender.</div>
          <div>You will then swap out words in the sentence that identify that group for a different demographic group, while keeping the rest of the sentence the same.
            This includes any pronouns or names that identify gender, ethnicity or age.
          </div>
          <p><b>Here are some examples of swaps:</b></p></div>
        <div className="examples">
          “Alice is cooking dinner for her children.”<br />
Demographic axis: Gender <br />
Rewrite: “Adam is cooking dinner for his children.” <br />
        </div>
        <div className="examples">
          <p>“The girl likes to play with her dolls all day.”</p>
          <p>Demographic axis: Gender</p>
          <p>Rewrite: “The boy likes to play with his dolls all day.”</p>
        </div>
        <div className="examples">
          <p>“The Asian engineer quickly codes up an algorithm.”</p>
          <p>Demographic axis: Ethnicity</p>
          <p>Rewrite: “The Black engineer quickly codes up an algorithm.”</p>
        </div>
        <div className="instructions">
          The resulting swap must identify a different group along the <b>same demographic axis</b>, so this is <b>NOT</b> a valid swap:
<div className="examples">“The Asian engineer quickly codes up an algorithm.” → “The happy engineer quickly codes up an algorithm.”</div>
        </div>
        <div>
          <p>When considering words to swap, try not to be influenced by your default assumptions! </p>

          <p>After completing your task, you will be asked to fill out a short demographic survey. Your responses are anonymous and will only be used in aggregate.</p>
        </div>
        <div className="reminders">{text}</div>
        <div>
          <b>IMPORTANT:</b> Messages you send in interacting with this bot will be used by the requestor and others for research purposes. This includes public disclosure of the messages as part of research data sets and research papers. Please ensure that your messages to the bot do not contain personal information, such as, for example, your name, address, email, or phone number, or other information that you would wish to keep private.
</div>
      </Directions>
      <section className="section">
        <div className="container">
          <p className="subtitle is-5">Swap words in this sentence to change the demographic axis from woman to man. <text className="red">Please swap <b>every</b> word that marks the female gender.</text></p>
          <p className="subtitle is-3 is-spaced">She cleans the dishes for her family every night.</p>
          <div className="field">
            <div className="field is-grouped">
              <p>Demographic Axis:</p>
              <div id="dropdown-box">
                <DropdownButton id="dropdown-basic-button" title={state}>
                  <Dropdown.Item href="#/ethnicity" onClick={() => handleUpdate("ethnicity")}>Ethnicity</Dropdown.Item>
                  <Dropdown.Item href="#/gender" onClick={() => handleUpdate("gender")}>Gender</Dropdown.Item>
                  <Dropdown.Item href="#/age" onClick={() => handleUpdate("age")}>Age</Dropdown.Item>
                  <Dropdown.Item href="#/none" onClick={() => handleUpdate("None")}>None</Dropdown.Item>
                </DropdownButton>
              </div>
            </div>
            <div>
              <p>Rewritten sentence:</p>
              <textarea type="text" style={{ minWidth: '50%' }} onInput={e => response.rewrite = e.target.value} />
            </div>
          </div>

          <p className="subtitle is-5">For each of the following examples, mark whether or not they are valid perturbations.</p>
          <div className="examples">
            <p>“The Asian engineer quickly codes up an algorithm.”</p>
            <p>Demographic axis: Ethnicity</p>
            <p>Rewrite: “The Black engineer quickly codes up an algorithm.”</p>
          </div>
          <div className="field is-grouped">
            <p>Is this valid?</p>
            <div>
              <form>
                <input type="radio" value="Yes" name="ethnicity" onClick={() => response.questions.ethnicity_perturb = "yes"} /> Yes
        <input type="radio" value="No" name="ethnicity" onClick={() => response.questions.ethnicity_perturb = "no"} /> No
        </form>
            </div>
          </div>
          <div className="examples">
            <p>“Frank who answered the phone said he was the store manager and he listened to my story.”</p>
            <p>Demographic axis: Gender</p>
            <p>Rewrite: “Frank who answered the phone said she was the store manager and he listened to my story.”</p>
          </div>
          <div className="field is-grouped">
            <p>Is this valid?</p>
            <div>
              <form>
                <input type="radio" value="Yes" name="gender" onClick={() => response.questions.gender_perturb = "yes"} /> Yes
        <input type="radio" value="No" name="gender" onClick={() => response.questions.gender_perturb = "no"} /> No
        </form>
            </div>
          </div>
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
      Demographic Survey
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
  // State variables
  const [showMoreText, setShowMoreText] = React.useState(false)
  const  [state,setState]=useState("Choose a demographic axis");
  const  [perturbed,setPerturbedState]=useState("");

  // Collecting demographic survey responses
  const [genderFields, setGenderFields] = useState({});
  const [raceFields, setRaceFields] = useState({});
  const [textFields, setTextFields] = useState({});

  const text = "Please write in clear English sentences that are free of grammatical errors.\n \
  Avoid racism, sexism, or other discriminatory or offensive language. \n \
  Do not provide any personal information in your sentences, such as your name, address, email or phone number. \n \
  Incomplete work will not be rewarded. \n \
  You must click “submit” when your work is complete.";
  const response = { "demographic": "", "rewrite": "", "survey": {"gender": [], "race": []}}
  function handleUpdate(demographic) {
      response.demographic = demographic
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
    response["demographic"] = state
    response["rewrite"] = perturbed
    console.log(response)
    onSubmit(response)
  }
  const showExpanded = true
  let additionalInstructions = (
    <div>
      <div><p><b>Here are some examples of swaps:</b></p></div>
      <div className="examples">
      “Alice is cooking dinner for her children.”<br />
Demographic axis: Gender <br />
Rewrite: “Adam is cooking dinner for his children.” <br />
      </div>
      <div className="examples">
        <p>“The girl likes to play with dolls all day.”</p>
        <p>Demographic axis: Gender</p>
        <p>Rewrite: “The boy likes to play with dolls all day.”</p>
      </div>
      <div className="examples">
        <p>“The Asian engineer quickly codes up an algorithm.”</p>
        <p>Demographic axis: Ethnicity</p>
        <p>Rewrite: “The Black engineer quickly codes up an algorithm.”</p>
      </div>
      <div className="instructions">
        The resulting swap must identify a different group along the <b>same demographic axis</b>, so this is <b>NOT</b> a valid swap:
<div className="examples">“The Asian engineer quickly codes up an algorithm.” → “The happy engineer quickly codes up an algorithm.”</div>
      </div>
      <div>
        <p>When considering words to swap, try not to be influenced by your default assumptions! </p>

        <p>After completing your task, you will be asked to fill out a short demographic survey. Your responses are anonymous and will only be used in aggregate.</p>
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
        <p className="title is-3 is-spaced">Rewrite sentences</p>
        <div className="instructions">The goal of this task is to rewrite sentences by swapping words or phrases along certain demographic axis while preserving the meaning of the sentence.
      <br />
          <div>You will be shown a sentence, and asked to identify whether the sentence contains references to any demographic group, such as ethnicity or gender.</div>
          <div>You will then swap out words in the sentence that identify that group for a different demographic group, while keeping the rest of the sentence the same.</div>
        </div>

        {showMoreText ? additionalInstructions : <button
          className="button"
          onClick={() => setShowMoreText(true)}
        >
          Show more
              </button>}

        <div className="reminders red">{text}</div>
        <div>
          <b>IMPORTANT:</b> Messages you send in interacting with this bot will be used by the requestor and others for research purposes. This includes public disclosure of the messages as part of research data sets and research papers. Please ensure that your messages to the bot do not contain personal information, such as, for example, your name, address, email, or phone number, or other information that you would wish to keep private.
</div>
      </Directions>
      <section className="section">
        <div className="container">
          <p className="subtitle is-5">Swap words in this sentence to change the demographic group.</p>
          <p className="subtitle is-3 is-spaced">{taskData.text}</p>
          <div className="field">
            <div className="field is-grouped">
              <p>Demographic Axis:</p>
              <div id="dropdown-box">
              <DropdownButton id="dropdown-basic-button" title={state}>
                  <Dropdown.Item href="#/ethnicity" onClick={() => handleUpdate("ethnicity")}>Ethnicity</Dropdown.Item>
                  <Dropdown.Item href="#/gender" onClick={() => handleUpdate("gender")}>Gender</Dropdown.Item>
                  <Dropdown.Item href="#/age" onClick={() => handleUpdate("age")}>Age</Dropdown.Item>
                  <Dropdown.Item href="#/none" onClick={() => handleUpdate("None")}>None</Dropdown.Item>
                </DropdownButton>
              </div>
            </div>
            <div>
              <p>Rewritten sentence:</p>
              <textarea 
              type="text" 
              style={{ minWidth: '50%' }} 
              id={`rewrite`}
              name="rewrite"
              onChange={e => setPerturbedState(e.target.value)} 
              />
            </div>
          </div>
          <DemographicSurvey 
          genderFields={genderFields} 
          setGenderFields={setGenderFields}
          raceFields={raceFields}
          setRaceFields={setRaceFields}
          textFields={textFields}
          setTextFields={setTextFields}
          >
        </DemographicSurvey>
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
