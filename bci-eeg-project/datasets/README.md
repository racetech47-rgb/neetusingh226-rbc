# PhysioNet EEG Motor Imagery Dataset

## What is it?

The **PhysioNet EEG Motor Movement/Imagery Dataset** (also known as the
*BCI2000* EEG dataset) contains EEG recordings from 109 subjects performing
real and imagined hand/feet movements.

- **Subjects**: 109
- **Channels**: 64 (10-20 system)
- **Sample rate**: 160 Hz
- **Tasks**: rest, left-hand imagery, right-hand imagery, feet, both hands
- **Format**: EDF (European Data Format)

Reference: Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
*Circulation*, 101(23), e215–e220.

---

## Download

```bash
python main.py --mode download-data
```

This will download subjects 1–3 by default into `datasets/physionet/`.

To download additional subjects, edit `datasets/physionet_loader.py`:

```python
download_physionet(subject_ids=list(range(1, 10)))
```

---

## License

The PhysioNet EEG Motor Imagery dataset is licensed under the
[Open Data Commons Attribution License (ODC-By)](https://opendatacommons.org/licenses/by/1.0/).

You are free to share and adapt the data, provided you give appropriate credit.

---

## Citation

If you use this dataset in your work, please cite:

```
Goldberger, A. L., Amaral, L. A., Glass, L., Hausdorff, J. M., Ivanov, P. C.,
Mark, R. G., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and
PhysioNet: components of a new research resource for complex physiologic signals.
Circulation, 101(23), e215–e220.

Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R.
(2004). BCI2000: a general-purpose brain-computer interface (BCI) system.
IEEE Transactions on Biomedical Engineering, 51(6), 1034–1043.
```
