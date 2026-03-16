before running BER_cal.mlx: 

move/copy the H_val_generated.mat file 
	at JMMD/ (or DANN/) 
	     model/ 
		GAN_cal_{settings}/   ({settings} is A300_B100_300 or D30_B100_300)
		   {SNR}_dB
		   	ver_{idx}
			    GAN_linear/ (or GAN_practical)
				H_visualize/
				    H_val_generated.mat
	(generated after running .py code
		JMMD/
		    code_{settings}/   ({settings} is A300_B100_300 or D30_B100_300)
		   	{mode}_{SNR}dB/    ({mode} is LI or Prac)
				JMMD_GAN_{SNR}dB_{mode}.py
	    or  Domain_Adversarial/
		    code_to_run/
			run_{settings}/   ({settings} is A300_B100_300 or D30_B100_300)
			     run_{SNR}dB_{mode}/    ({mode} is LI or Prac)
				 UDA_GAN_v4_11_{SNR}dB_{mode}.py
	)
to the corresponding folder {settings}_{model}/     ({settings} is A300_B100_300 or D30_B100_300
						     {model}    is DANN or JMMD )
and rename to {mode}_GAN_{SNR}dB.mat     ({mode} is LI or Prac)