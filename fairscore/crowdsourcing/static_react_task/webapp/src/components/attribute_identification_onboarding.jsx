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
import ToggleButton from 'react-bootstrap/ToggleButton'
import ToggleButtonGroup from 'react-bootstrap/ToggleButtonGroup'
const taskData = {
    "text": "Every night, Jessica prepares some recipes for her toddlers that she learned from her Japanese relatives .", 
    "spans": {
        "gender": [["Jessica", 2], ["her", 7], ["she", 10], ["her", 13]], 
        "race": [["for", 6], ["Japanese", 14]], 
        "age": [["toddlers", 8]]
    }
}

function AttributeIdentificationOnboardingComponent({ onSubmit }) {
    // State variables
    const [showMoreText, setShowMoreText] = React.useState(false)
    const [state,setState]=useState("Select a demographic group");
    const [perturbed,setPerturbedState]=useState("");
    // Track all the words and indices
    const wordMapping = {}
    for (const key of Object.keys(taskData["spans"])) {
      console.log(key)
      wordMapping[key] = {}
      for (const [attribute, idx] of taskData["spans"][key]) {
        console.log(attribute)
        wordMapping[key][attribute] = {}
        console.log(wordMapping)
      }
    }
    const [wordMap, setWordMapState] = useState(wordMapping);
    console.log(wordMap)
  
    // Collecting demographic survey responses
    const [genderFields, setGenderFields] = useState({});
    const [raceFields, setRaceFields] = useState({});
    const [textFields, setTextFields] = useState({});
  
    const text = "Please write in clear English sentences that are free of grammatical errors.\n \
    Avoid racism, sexism, or other discriminatory or offensive language. \n \
    Do not provide any personal information in your sentences, such as your name, address, email or phone number. \n \
    Incomplete work will not be rewarded. \n \
    You must click “submit” when your work is complete.";
    const response = { "survey": {"gender": [], "race": []} }
    function handleUpdate(demographic, axis, word, idx) {
        response.demographic = demographic
        setState(demographic);
        const wordMapGender = {...wordMap[axis]}
        wordMapGender[word][idx] = demographic
        setWordMapState({...wordMap, [axis] : wordMapGender});
        console.log(wordMap);
    }
    // Enumerate types
    const attributeMap = {
      "gender": {
        "man": "Man",
        "woman": "Woman",
        "non-binary": "Non-Binary",
        "none": "None or Unspecified",
      },
      "race": {
        "hispanic": "Hispanic or Latino",
        "pacific-islander": "Native Hawaiian or Other Pacific Islander",
        "white": "White",
        "black": "Black or African American",
        "native-american": "American Indian or Alaska Native",
        "asian": "Asian",
        "none": "None or Unspecified",
      },
      "age": {
        "child": "Child (< 18)",
        "young": "Young (18-44)",
        "middle-aged": "Middle-aged (45-64)",
        "old": "Senior (65+)",
        "adult": "Adult (unspecified)",
        "none": "None or Unspecified",
      }
    }
  
    function RenderDemographicAttributeDropDown(span, axis, i) {
      const [word, idx] = span
      return (
        <div className="annotate-words">
        <text className="span-index">{i + 1}</text>
        <text className="demographic">{word}</text>
        <div className="dropdown-unit">
          <DropdownButton id="dropdown-basic-button" className="dropdown" title={wordMap[axis][word][idx] ? attributeMap[axis][wordMap[axis][word][idx]] : "Select a demographic group"}>
          {Object.keys(attributeMap[axis]).map((attribute) => <Dropdown.Item className="dropdown" onClick={() => handleUpdate(attribute, axis, word, idx)}>{attributeMap[axis][attribute]}</Dropdown.Item>)}
          </DropdownButton>
        </div>
        </div>
      );
    }
  
    function RenderSnippet(text, words) {
      // text split into array of words
      const textArray = text.split(" ")
      words.forEach(function(word_pair, i) {
          const [word, idx] = word_pair
          const renderIndex = i + 1
          textArray[idx] = "<text id='highlight'>" + word + "</text>" + "<text id='superscript'>" + renderIndex + "</text>" 
      })
      return textArray.join(" ")
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
      response["spans"] = wordMap
      console.log(response)
      onSubmit(response)
    }
    const showExpanded = true
  
    return (
      <div>
        <Directions>
          <p className="title is-3 is-spaced">Identify Attributes</p>
          <div className="instructions">The goal of this task is to identify demographic groups in highlighted words in a sentence.
        <br />
            <div>You will be shown a sentence with highlighted words, and asked to identify which demographic group the word refers to, such as ethnicity or gender.</div>
            <div>If there is no demographic group, select "None or Unspecified".</div>
          </div>
          <div>
            <div><p><b>Here are some examples of attributes and their demographic groups:</b></p></div>
            <div id="examples" className="examples">
            “<text id="highlight">Alice</text> is cooking dinner for <text id="highlight">her</text> children.”<br />
      Demographic axis: <b>Gender</b> <br />
      <text className="red">Alice</text>: Woman <br />
      <text className="red">her</text>: Woman <br />
            </div>
            <div className="examples">
              <p>“The <text id="highlight">Asian</text> engineer quickly codes up an algorithm.”</p>
              <p>Demographic axis: <b>Ethnicity</b></p>
              <p><text className="red">Asian</text>: Asian</p>
            </div>
            <div className="examples">
              <p>“The <text id="highlight">young</text> man was worried that he was running late to work.”</p>
              <p>Demographic axis: <b>Age</b></p>
              <p><text className="red">young</text>: young (18-44)</p>
            </div>
            <div>
              <p>If an age reference can fit multiple >18 categories, select "adult, unspecified". </p>
      
              <p>After completing your task, you will be asked to fill out a short demographic survey. Your responses are anonymous and will only be used in aggregate.</p>
            </div>
            <button
              className="button"
              onClick={() => setShowMoreText(false)}
            >
              Show less
                    </button>
          </div>
  
          <div className="reminders red">{text}</div>
          <div>
            <b>IMPORTANT:</b> Messages you send in interacting with this bot will be used by the requestor and others for research purposes. This includes public disclosure of the messages as part of research data sets and research papers. Please ensure that your messages to the bot do not contain personal information, such as, for example, your name, address, email, or phone number, or other information that you would wish to keep private.
  </div>
        </Directions>
        <section className="section">
          <div className="container">
            <div id="test"></div>
            <p className="subtitle is-5">For each highlighted word in this sentence, select the demographic group identified, or "None" if there is none.</p>
            
            <p className="subtitle margin-top">Please categorize the following <text className="red"><b>gender</b></text> referring words, or select "None or Unspecified".</p>
            <p className="subtitle is-3 is-spaced"><div dangerouslySetInnerHTML={{ __html: RenderSnippet(taskData.text, taskData["spans"]["gender"]) }}></div></p>
            <div className="field">
              <div className="field is-grouped">
                <div>{taskData["spans"]["gender"].map((span, i) => RenderDemographicAttributeDropDown(span, "gender", i))}</div>
              </div>
            </div>
  
            <p className="subtitle margin-top">Please categorize the following <text className="red"><b>race/ethnicity</b></text> referring words, or select "None or Unspecified".</p>
            <p className="subtitle is-3 is-spaced"><div dangerouslySetInnerHTML={{ __html: RenderSnippet(taskData.text, taskData["spans"]["race"]) }}></div></p>
            <div className="field">
              <div className="field is-grouped">
                <div>{taskData["spans"]["race"].map((span, i) => RenderDemographicAttributeDropDown(span, "race", i))}</div>
              </div>
            </div>
  
            <p className="subtitle margin-top">Please categorize the following <text className="red"><b>age</b></text> referring words, or select "None or Unspecified".</p>
            <p className="subtitle is-3 is-spaced"><div dangerouslySetInnerHTML={{ __html: RenderSnippet(taskData.text, taskData["spans"]["age"]) }}></div></p>
            <div className="field">
              <div className="field is-grouped">
                <div>{taskData["spans"]["age"].map((span, i) => RenderDemographicAttributeDropDown(span, "age", i))}</div>
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

export { AttributeIdentificationOnboardingComponent as AttributeIdentificationOnboardingComponent };