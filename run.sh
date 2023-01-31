export QUOTA_VARIANT='single'
export LEFT=1
export COMPANY_N=100
export WORKER_DS_N=300
export WORKER_SDE_N=300
export COMPANY_DS_QUOTA=1
export COMPANY_SDE_QUOTA=1
export T=3000
export REPLICATION=2
export SUB_TYPE=True

nohup python3 main.py \
    --quota_variant=$QUOTA_VARIANT \
    --left=$LEFT \
    --company_n=$COMPANY_N \
    --worker_ds_n=$WORKER_DS_N \
    --worker_sde_n=$WORKER_SDE_N \
    --company_ds_quota=$COMPANY_DS_QUOTA \
    --company_sde_quota=$COMPANY_SDE_QUOTA \
    --T=$T \
    --rep=$REPLICATION \
    --sub_type=$SUB_TYPE \
>../log/log_$(date +"%Y_%m_%d_%I_%M_%p")_{$QUOTA_VARIANT}_{$LEFT}_{$COMPANY_N}_{$WORKER_DS_N}_{$WORKER_SDE_N}_{$T}_{$COMPANY_DS_QUOTA}_{$COMPANY_SDE_QUOTA}_{$REPLICATION}_{$SUB_TYPE}.txt &



