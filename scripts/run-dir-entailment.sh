for f in $1/*
do
    sbatch scripts/submit-entailment.sh $f
done

echo 'Done!'