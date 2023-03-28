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

export { OnboardingComponent as OnboardingComponent };