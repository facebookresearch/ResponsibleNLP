# FairScore React Task

NOTE: This README is WIP.

To run this data collection pipeline, first make sure you are running python 3.7+.

`conda create -n fairscore_env python==3.7.4 pip`

Install Mephisto via these instructions: https://github.com/facebookresearch/mephisto/blob/master/docs/quickstart.md

To run a crowdsourcing task with the local architect:
`python run_task.py conf=onboarding_example`

If testing local UI changes, we recommend in a separate terminal, running
`npm run dev:watch`

This ensures your react changes are updated in the mephisto task in real time.