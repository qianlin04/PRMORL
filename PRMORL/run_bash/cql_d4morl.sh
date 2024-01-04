for dataset_type in amateur_uniform expert_uniform
do
    use_wandb=1
    policy_freq=2           
    eval_freq=1000000 
    time_steps=500000 #temp
    pref_gen_way='L1_return'
    

    for env in Hopper Walker2d Swimmer Ant HalfCheetah 
    do
        seed=4107

        if [ "$env" = "Hopper" ]; then
            gpu=0
            conservative_weight=5.0
        elif [ "$env" = "Ant" ]; then
            gpu=2
            conservative_weight=2.0
        elif [ "$env" = "Swimmer" ]; then
            gpu=4
            conservative_weight=1.0
        elif [ "$env" = "Walker2d" ]; then
            gpu=6
            if [ "$dataset_type" = "expert_uniform" ]; then
                conservative_weight=10.0
            else
                conservative_weight=20.0
            fi
        elif [ "$env" = "HalfCheetah" ]; then
            gpu=8
            conservative_weight=10.0
        fi

        algo='CQL'
        weight_num=1
        normalize_states=1
        lr_decay=1
        pref_perturb_theta=1.0
        nohup python train_Mujoco_MOOF_TD3_HER.py \
            --algo $algo \
            --env $env --dataset_type $dataset_type \
            --seed $seed \
            --dataset 'd4morl' \
            --gpu $gpu \
            --use_wandb $use_wandb \
            --policy_freq $policy_freq \
            --num_objective 2 \
            --pref_perturb_theta $pref_perturb_theta \
            --weight_num $weight_num \
            --eval_freq $eval_freq\
            --time_steps $time_steps\
            --pref_gen_way $pref_gen_way\
            --normalize_states $normalize_states\
            --lr_decay $lr_decay\
            --conservative_weight $conservative_weight\
            > logs/${env}_${dataset_type}_${seed}.txt &
        seed=$(($seed+1))
        gpu=$(($gpu+1))

        nohup python train_Mujoco_MOOF_TD3_HER.py \
            --algo $algo \
            --env $env --dataset_type $dataset_type \
            --seed $seed \
            --dataset 'd4morl' \
            --gpu $gpu \
            --use_wandb $use_wandb \
            --policy_freq $policy_freq \
            --num_objective 2 \
            --pref_perturb_theta $pref_perturb_theta \
            --weight_num $weight_num \
            --eval_freq $eval_freq\
            --time_steps $time_steps\
            --pref_gen_way $pref_gen_way\
            --normalize_states $normalize_states\
            --lr_decay $lr_decay\
            --conservative_weight $conservative_weight\
            > logs/${env}_${dataset_type}_${seed}.txt &
        seed=$(($seed+1))
        gpu=$(($gpu+1))

    done
done    