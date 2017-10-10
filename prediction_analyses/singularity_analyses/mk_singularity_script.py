njobs=8
container='/work/01329/poldrack/stampede2/singularity_images/selfregulation-2017-10-10-646e0b351ab0.img'
for t in ['baseline','task','survey']:
  with open('%s_sing.sh'%t,'w') as f:
    for i in range(100):
       f.write("singularity run -e %s /workdir/Self_Regulation_Ontology/prediction_analyses/behav_prediction.py -d %s -r /scratch/01329/poldrack/SRO -j %d\n"%(container,t,njobs)) 
