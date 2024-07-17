
First, it is important to work in a Conda venv. This should be the venv where you installed suite2p. Activate it with conda activate <venv_name>
Then, in the directory for each organism, there should be a yaml file in the following format:

name: DON-012207
dob: '20230101'
sex: 'male'
cohort_year:
- 2023
pdays:
- 60

session_ids:
- 20240522
- 20240523

session_names:
- 20240522
- 20240523

In each session directory, there must be a <session_id>.yaml file.
Its structure should be the following: 

animal_id: DON-011733
date: 20240522
daytime: ''
method: 2P
notes: ''
rot_angle: -0.3
rot_center_yx:
- 57.2
- 385.4
updated: true
session_type: 'day0'
weight: ''
y_shift: -4
x_shift: -2

Importantly, for each organism there must be exactly one yaml file where the session_type is 'day0'.
Additionally, it is necessary to have one rois_pixels_and_thresholds_<day0>.npz file where <day0> is only added for the one day0 session.
This should be the output of BMI experiments. If it does not exist yet, it has to be generated from .mesc or .raw files via the preprocessing script. 
Check for the filetype and either start after the .mesc to .raw conversion or before. 