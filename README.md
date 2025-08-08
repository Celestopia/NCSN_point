Implementation of NCSNv2 on 2-d point dataset.

Run `run.py` to conduct a single experiment with command line arguments.

Example:
```bash
python run.py --k_i 0.1 --k_d 6.0
```

Run `run2.py` to conduct multiple experiments. You can customize the experiment loops by modifying `run2.py`.

`main_light.py` is a light-weight version of `main.py`. You can customize the experiment loops in `run2_light.py` to run efficient experiments.

All results will be saved in `results` directory.

For more details, please start from `main.py`. The structure and logic of this project are well documented in the comments.
