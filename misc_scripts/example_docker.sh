docker run --rm  \
--mount type=bind,src=/media/Data/Ian/Experiments/expfactory/Self_Regulation_Ontology/Data,dst=/SRO/Data \
--mount type=bind,src=/media/Data/Ian/Experiments/expfactory/Self_Regulation_Ontology/batch_files/container_scripts/output,dst=/output \
--entrypoint /bin/bash \
-ti sro