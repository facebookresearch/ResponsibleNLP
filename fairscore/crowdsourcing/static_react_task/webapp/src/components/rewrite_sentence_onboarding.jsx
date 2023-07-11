/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React, { useState } from "react";
import { DropdownButton, Dropdown, Form, ButtonGroup } from "react-bootstrap";
import { OnboardingComponent } from "./onboarding.jsx"

/* Task information for onboarding questions. First task checks that workers are
 * able to change "king" to "queen", change the name "James", and keep the rest of the sentence
 * the same. The second task checks that workers perturb the pronouns to terms that do not
 * refer to binary genders.
 */
const taskData1 = {
    "text": "king james has a daughter with his second wife.",
    "demographic_axis": "gender",
    "selected_word": "king",
    "selected_word_index": 0,
    "selected_word_category": "male",
    "perturbed_word_category": "female",
}

const taskData2 = {
  "text": "He went to the mall by himself and bought some video games.",
  "demographic_axis": "gender",
  "selected_word": "He",
  "selected_word_index": 0,
  "selected_word_category": "male",
  "perturbed_word_category": "non-binary",
}

const taskData3 = {
  "text": "People came to copy old photos of loved ones.",
  "demographic_axis": "age",
  "selected_word": "old",
  "selected_word_index": 4,
  "selected_word_category": "senior (65+)",
  "perturbed_word_category": "young (18-44)",
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

function RewriteSentenceOnboardingComponent({ onSubmit }) {
  // State variables
  const [showMoreText, setShowMoreText] = React.useState(false)
  const  [state,setState]=useState("Choose a demographic axis");
  const  [q1Answer,setQ1State]=useState("");
  const  [q2Answer,setQ2State]=useState("");
  const  [q3Answer,setQ3State]=useState("");

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
  // Counter for submission attempts
  let numAttempts = 0
  function handleUpdate(demographic) {
      response.demographic = demographic
      setState(demographic);
  }

  function RenderSnippet(text, word_index) {
    // text split into array of words
    const textArray = text.split(" ")
    textArray[word_index] = "<text id='highlight'>" + textArray[word_index] + "</text>"
    return textArray.join(" ")
  }

  function handleSubmit() {
    // If number of attempts is within threshold, add leniency checks
    if (numAttempts <= 2) {
      // If string is all whitespace, ask to complete the task
      if (!/\S/.test(q1Answer) || !/\S/.test(q2Answer) || !/\S/.test(q3Answer)) {
        alert("Please complete the task before submitting.")
        numAttempts++
        return
      }
      // If string is the same, ask to complete the task
      if (q1Answer == taskData1.text || q2Answer == taskData2.text) {
        alert("Please rewrite the sentence.")
        numAttempts++
        return
      }
      // Send warning if there isn't complete punctuation
      if (!q1Answer.includes(".") || !q2Answer.includes(".")) {
        alert("Please use full puncutation.")
        numAttempts++
        return
      }
      // Check for common onboarding errors, eg. Mary's wife -> Mary's husband
      if (q1Answer.includes("husband")) {
        alert("Please try again.")
        numAttempts++
        return
      }
      // The second question checks for non-binary pronouns
      if (q2Answer.includes("She") || q2Answer.includes("herself")) {
        alert("Please try again.")
        numAttempts++
        return
      }
      // The third question highlights a word that doesn't contain demographic info
      // We expect the rewrite to be the same as the original
      if (!q3Answer.includes("old photos")) {
        alert("Please try again. If the highlighted word does not contain demographic information about a person, write the original sentence.")
        numAttempts++
        return
      }
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
    response["demographic"] = state
    response["q1_rewrite"] = q1Answer
    response["q2_rewrite"] = q2Answer
    response["q3_rewrite"] = q3Answer
    console.log(response)
    onSubmit(response)
  }

  console.log(taskData1)

  return (
    <div>
      <Directions>
        <p className="title is-3 is-spaced">Rewrite sentences</p>
        <div className="instructions">The goal of this task is to rewrite sentences by changing words or phrases along certain demographic axis while preserving the meaning of the sentence.
      <br />
          <div>You will be shown a sentence, and you will be given a word in the sentence that contains demographic information about a person(s), such as ethnicity, gender or age.</div>
          <div>You will then change words in the sentence that identify that person(s) to refer to a different demographic group, while ensuring fluency and grammatical correctness.</div>
          <div>If the highlighted word does not refer to a person, please write the text as is.</div>
        </div>
        <div>
            <div><p><b>Here are some examples of swaps:</b></p></div>
            <div className="examples">
            “<text id="highlight">Alice</text> is cooking dinner for her children.”<br />
            Demographic axis: <text className="red">gender</text> <br />
            Instruction: <i>Please change the word <text id="highlight">Alice</text> and all references to <text id="highlight">Alice</text> from <text className="red">female</text> to <text className="red">male</text>. </i><br />
            Rewrite: “<text id="highlight">Adam</text> is cooking dinner for <text id="highlight">his</text> children.” <br />
            </div>
            <div className="examples">
                <p>“The <text id="highlight">girl</text> likes to play with dolls all day.”</p>
                <p>Demographic axis: <text className="red">gender</text></p>
                Instruction: <i>Please change the word <text id="highlight">girl</text> and all references to <text id="highlight">girl</text> from <text className="red">female</text> to <text className="red">male</text>. </i><br />
                <p>Rewrite: “The <text id="highlight">boy</text> likes to play with dolls all day.”</p>
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
        </div>
        <div className="reminders red">{text}</div>
        <div>
          <b>IMPORTANT:</b> Your annotations will be used by the requestor and others for research purposes. This includes public disclosure of the work as part of research data sets and research papers. Please ensure that your work does not contain personal information, such as, for example, your name, address, email, or phone number, or other information that you would wish to keep private.
</div>
      </Directions>
      <section className="section">
        <div className="container">
          <p className="subtitle is-5">Rewrite this sentence to change the demographic group for the highlighted word, while keeping the meaning as close to the original as possible (while ensuring fluency and grammatical correctness).</p>
          <p className="subtitle is-3 is-spaced"><div dangerouslySetInnerHTML={{ __html: RenderSnippet(taskData1.text, taskData1["selected_word_index"]) }}></div></p>
          <div className="field">
            <div className="field">
              <p className="font-size-med"><b>Demographic Axis: <text className="red">{taskData1["demographic_axis"]}</text></b></p>
            </div>
            <div className="field">
              <p className="font-size-med"><b>Please change the word <text id="highlight">{taskData1["selected_word"]}</text> and all references to <text id="highlight">{taskData1.selected_word}</text> from <text className="red">{taskData1.selected_word_category}</text> to <text className="red">{taskData1["perturbed_word_category"]}</text>, including names if appropriate.</b> If the highlighted word does not refer to a person, please write the text as is.</p>
            </div>
            <div>
              <p className="font-size-med">Rewritten sentence:</p>
              <textarea 
              className="font-size-small"
              type="text" 
              rows="8"
              style={{ minWidth: '90%' }} 
              id={`rewrite`}
              name="rewrite"
              onChange={e => setQ1State(e.target.value)} 
              />
            </div>
          </div>

          <p className="subtitle is-3 is-spaced"><div dangerouslySetInnerHTML={{ __html: RenderSnippet(taskData2.text, taskData2["selected_word_index"]) }}></div></p>
          <div className="field">
            <div className="field">
              <p className="font-size-med"><b>Demographic Axis: <text className="red">{taskData2["demographic_axis"]}</text></b></p>
            </div>
            <div className="field">
              <p className="font-size-med"><b>Please change the word <text id="highlight">{taskData2["selected_word"]}</text> and all references to <text id="highlight">{taskData2.selected_word}</text> from <text className="red">{taskData2.selected_word_category}</text> to <text className="red">{taskData2["perturbed_word_category"]}</text>, including names if appropriate.</b> If the highlighted word does not refer to a person, please write the text as is.</p>
            </div>
            <div>
              <p className="font-size-med">Rewritten sentence:</p>
              <textarea 
              className="font-size-small"
              type="text" 
              rows="8"
              style={{ minWidth: '90%' }} 
              id={`rewrite`}
              name="rewrite"
              onChange={e => setQ2State(e.target.value)} 
              />
            </div>
          </div>

          <p className="subtitle is-3 is-spaced"><div dangerouslySetInnerHTML={{ __html: RenderSnippet(taskData3.text, taskData3["selected_word_index"]) }}></div></p>
          <div className="field">
            <div className="field">
              <p className="font-size-med"><b>Demographic Axis: <text className="red">{taskData3["demographic_axis"]}</text></b></p>
            </div>
            <div className="field">
              <p className="font-size-med"><b>Please change the word <text id="highlight">{taskData3["selected_word"]}</text> and all references to <text id="highlight">{taskData3.selected_word}</text> from <text className="red">{taskData3.selected_word_category}</text> to <text className="red">{taskData3["perturbed_word_category"]}</text>, including names if appropriate.</b> If the highlighted word does not refer to a person, please write the sentence as is.</p>
            </div>
            <div>
              <p className="font-size-med">Rewritten sentence:</p>
              <textarea 
              className="font-size-small"
              type="text" 
              rows="8"
              style={{ minWidth: '90%' }} 
              id={`rewrite`}
              name="rewrite"
              onChange={e => setQ3State(e.target.value)} 
              />
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

export { RewriteSentenceOnboardingComponent as OnboardingComponent };