"""
Use latent data to compute the std of your data, then use the std to normalize the data
by (1/std) to get scaled inputs for Diffusion Transformer

(for instance, with riffusion, the latent scaling factor is 0.185)

"""


#TODO