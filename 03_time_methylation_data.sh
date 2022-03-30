# A master script to run the dataset from the Handl et al paper,
# used to make figure 0, the barplot used in the poster. Warning:
# The wenda_orig command will become unresponsive after several
# hours, so you may need to kill it manually using htop.

bash time_wenda_gpu.sh handl 100
bash time_wenda_orig.sh handl
