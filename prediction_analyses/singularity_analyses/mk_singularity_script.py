with open('baseline_sing.sh','w') as f:
    for i in range(100):
       f.write("singularity run -e /work/01329/poldrack/stampede2/singularity_images/selfregulation-2017-10-09-6c180963393a.img /workdir/Self_Regulation_Ontology/prediction_analyses/behav_prediction.py -d baseline -r /scratch/01329/poldrack/SRO -j 8\n") 
with open('survey_sing.sh','w') as f:
    for i in range(100):
       f.write("singularity run -e /work/01329/poldrack/stampede2/singularity_images/selfregulation-2017-10-09-6c180963393a.img /workdir/Self_Regulation_Ontology/prediction_analyses/behav_prediction.py -d survey -r /scratch/01329/poldrack/SRO -j 8\n") 
with open('task_sing.sh','w') as f:
    for i in range(100):
       f.write("singularity run -e /work/01329/poldrack/stampede2/singularity_images/selfregulation-2017-10-09-6c180963393a.img /workdir/Self_Regulation_Ontology/prediction_analyses/behav_prediction.py -d task -r /scratch/01329/poldrack/SRO -j 8\n") 
