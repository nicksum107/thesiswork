#!/bin/bash

usage() {
    echo "Usage: ${0} [-a] [-i ITER] [-p PROCS] [-g GPUS] [-m MEM] [-t TIME] [-l THREADS] [-d DEPENDENCY] [-s SUFFIX] CONFIG" 1>&2;
    exit 1;
}

declare mem=16
declare iteration=0
declare walltime="06-00:00:00"
declare gpus=4
declare threads=4
declare dependency=""
declare all=false

while getopts ":al:i:p:d:g:m:t:s:" op; do
    case "${op}" in
        i) iteration=${OPTARG};;
        p) ppn=${OPTARG};;
        g) gpus=${OPTARG};;
        m) mem=${OPTARG};;
        t) walltime=${OPTARG};;
        l) threads=${OPTARG};;
        s) suffix=${OPTARG};;
        d) dependency=${OPTARG};;
        a) all=true;;
        \?) echo "Invalid option: -${OPTARG}" >&2
            exit -1;;
    esac
done

declare mem=$((mem * 1024))

if [ -z ${ppn} ]; then
    declare ppn=$((gpus * 4))
fi

shift $((OPTIND - 1))
if [ -z "${1}" ]; then
    usage
    exit -1
fi

if [ -z "${dependency}" ] && [ "${all}" == "false" ]; then
    sbatch \
        --mail-user=${USER}@cs.princeton.edu \
        --cpus-per-task=${ppn} \
        --mem=${mem} \
        --gres=gpu:${gpus} \
        --time=${walltime} \
        --chdir=${PWD} \
        --job-name=${1}-${suffix} \
        --account=pvl \
        --exclude=node[501,502] \
        ./slurm/config.sh ${1} ${iteration} ${threads} ${suffix}
elif [ -z "${dependency}" ] && [ "${all}" == "true" ]; then
    sbatch \
        --mail-user=${USER}@cs.princeton.edu \
        --cpus-per-task=${ppn} \
        --mem=${mem} \
        --gres=gpu:${gpus} \
        --time=${walltime} \
        --chdir=${PWD} \
        --job-name=${1}-${suffix} \
        --account=pvl \
        --exclude=node[501,502] \
        ./slurm/config.sh ${1} ${iteration} ${threads} ${suffix}
elif [ "${all}" == "false" ]; then
    sbatch \
        --mail-user=${USER}@cs.princeton.edu \
        --dependency=afterany:${dependency} \
        --cpus-per-task=${ppn} \
        --mem=${mem} \
        --gres=gpu:${gpus} \
        --time=${walltime} \
        --chdir=${PWD} \
        --job-name=${1}-${suffix} \
        --account=pvl \
        --exclude=node[501,502] \
        ./slurm/config.sh ${1} ${iteration} ${threads} ${suffix}
else
    sbatch \
        --mail-user=${USER}@cs.princeton.edu \
        --dependency=afterany:${dependency} \
        --cpus-per-task=${ppn} \
        --mem=${mem} \
        --gres=gpu:${gpus} \
        --time=${walltime} \
        --chdir=${PWD} \
        --job-name=${1}-${suffix} \
        --account=pvl \
        --exclude=node[501,502] \
        ./slurm/config.sh ${1} ${iteration} ${threads} ${suffix}
fi
