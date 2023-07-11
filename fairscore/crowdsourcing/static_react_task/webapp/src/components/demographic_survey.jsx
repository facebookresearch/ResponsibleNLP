/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React, { useState } from "react";
import { Dropdown, Form } from "react-bootstrap";


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

  export { DemographicSurvey as DemographicSurvey };