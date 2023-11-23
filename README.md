# binarize2pcalcium
Binarization pipelines for converting continuous 2P calcium imaging traces into binarized time series

# Required .yaml files

You will need 2 .yaml files to read the meta data of your recordings

ANIMAL_ID.yaml  <- inside the animal directory

SESSION_ID.yaml <- inside the session directory

Please see example .yaml files provided.

# Code basic use

```
from binarize2pcalcium import binarize2pcalcium as binca

data_dir = '/media/cat/2pdata'
animal_id = 'DON-011733'
session = '20230203'

c = binca.Calcium(data_dir, animal_id)

c.session = session 
c.session_name = session

c.data_type = '2p'
c.remove_bad_cells = False
c.verbose = False                          # outputs additional information during processing
c.recompute_binarization = True           # recomputes binarization and other processing steps; False: loads from previous saved locations

# set flags to save matlab and python data
c.save_python = True         # save output as .npz file 
c.save_matlab = False         # save output as .mat file

# manual thresholds for spike detection
c.dff_min = 0.05                  # min %DFF for [ca] burst to considered a spike (default 5%) overwrites percentile threshold parameter
c.percentile_threshold = 0.9999   # this is pretty fixed, we don't change it; we want [ca] bursts that are well outside the "physics-caused"noise
c.maximum_std_of_signal = 0.08     # if std of signal is greater than this, then we have a noisy signal and we don't want to binarize it
									  #    - this is a very important flag! come see me if you don't understand it

#
c.binarize_data()
```

