for f in $1/*
do
    sbatch scripts/submit-job.sh $f
done

echo 'Done!'