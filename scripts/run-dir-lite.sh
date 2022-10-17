for f in $1/*
do
    sbatch scripts/submit-job-lite.sh $f
done

echo 'Done!'