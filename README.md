# GPBO Web Application

Gaussian Process Bayesian optimization for material property optimization.
This project contains a nicegui-based webapp providing GPBO.

## Development
Install requirements using
```bash
$ python3 -m virtualenv venv
$ source ./venv/bin/activate
$ pip install -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/cpu
```

## Running
run using
```bash
python3 ./app.py
```

## Roadmap
- [x] move to nicegui
- [x] containerization and deployment on iai kubernetes cluster
- [x] tutorial on expected data format
- [x] improve layout
- [x] generalization to different input and output combinations 
- [x] add equality constraints on the inputs
- [x] removed XRD processing
- [x] for some reason, the app cannot complete sampling when I disable reloading -> fix
- [x] gather feedback from ben + simon and integrate
    - [x] add nonlinear inequality constraints
    - [x] determine constraint bug issue
    - [x] add logging during optimization
    - [x] allow zeroing of some constraints (n out of k) 
- [ ] optional XRD processing
- [ ] display error on sample_id column matching error 
- [x] add "preparing optimization" after clicking "Optimize while Optimize tab loads"
- [x] deal with fix n of k constraint category selection issues
    - samples may fall into multiple categories 
    - if more than n values accidentally match their fixation values
    - should these items belong to multiple or no categories?
- [ ] display model fit on optimization screen
