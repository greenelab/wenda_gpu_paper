# At this point, the wenda_gpu feature models should all be trained and
# confidence scores obtained, so we can now plug those scores into our
# ultimate weighted elastic net model. The output is used to make figures
# 1B-D. Note that since we were unable to run the full handl dataset on 
# cpu, we obtained the compiled confidence scores from the original authors
# of the Handl et al paper to input into their code.

python3 train_elastic_net.py -p handl --horvath 
python3 wenda_orig/elastic_net.py handl
