nid=0
modelid="model$nid"
submissionid="submission$nid"
now=$(date +"%T")
echo "Start time : $now"

th main.lua -modelName $modelid -augmentedData -cuda
th test_net.lua -modelName $modelid  -submissionName $submissionid -cuda
#cp $modelid.mean /home/mario/Dropbox/DSG/
#cp $modelid.net /home/mario/Dropbox/DSG/
#cp $modelid.stdv /home/mario/Dropbox/DSG/
#cp $submissionid.csv /home/mario/Dropbox/DSG/

now=$(date +"%T")
echo "End time : $now"

#th test_avg_pred.lua -submissionName submission_20_21_pavg -cuda
#cp submission_20_21_pavg.csv /home/mario/Dropbox/DSG/
#cp submission_20_21_pavg_detailed.csv /home/mario/Dropbox/DSG/
#th test_avg_weights.lua -submissionName submission_20_21_wavg -cuda
#cp submission_20_21_wavg.csv /home/mario/Dropbox/DSG/
#cp submission_20_21_wavg_detailed.csv /home/mario/Dropbox/DSG/
