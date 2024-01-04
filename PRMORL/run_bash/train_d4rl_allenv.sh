for dataset_type in medium-replay medium medium-expert 
do
    use_wandb=1
    w_bc_min=0.20    
    policy_freq=2           
    eval_way="mc|exhaust" 
    eval_freq=1000000
    time_steps=1000000
    pref_gen_way='L1_return'
    

    for env in Hopper Walker2d HalfCheetah 
    do
        seed=3002
        if [ "$env" = "Hopper" ]; then
            gpu=0
            weight_bc_reward=0.5 #1.0
        elif [ "$env" = "Walker2d" ]; then
            gpu=3
            weight_bc_reward=0.25 #0.5
        elif [ "$env" = "HalfCheetah" ]; then
            gpu=6
            weight_bc_reward=0.01 #0.05
        fi

        policy_regularization='Diffusion-QL'
        weight_num=1
        pref_perturb_theta=1.00
        diffusion_n_timesteps=5
        nohup python train_Mujoco_MOOF_TD3_HER.py \
            --env $env --dataset_type $dataset_type \
            --seed $seed \
            --dataset 'd4rl' \
            --gpu $gpu \
            --use_wandb $use_wandb \
            --weight_bc_reward $weight_bc_reward \
            --w_bc_min $w_bc_min \
            --policy_freq $policy_freq \
            --num_objective 2 \
            --eval_way $eval_way\
            --pref_perturb_theta $pref_perturb_theta \
            --weight_num $weight_num \
            --policy_regularization $policy_regularization\
            --diffusion_n_timesteps $diffusion_n_timesteps\
            --eval_freq $eval_freq\
            --time_steps $time_steps\
            --pref_gen_way $pref_gen_way\
            > logs/${env}_${dataset_type}_${seed}.txt &
        seed=$(($seed+1))
        gpu=$(($gpu+1))

        policy_regularization='TD3+BC'
        weight_num=1
        pref_perturb_theta=1.00
        nohup python train_Mujoco_MOOF_TD3_HER.py \
            --env $env --dataset_type $dataset_type \
            --seed $seed \
            --dataset 'd4rl' \
            --gpu $gpu \
            --use_wandb $use_wandb \
            --weight_bc_reward $weight_bc_reward \
            --w_bc_min $w_bc_min \
            --policy_freq $policy_freq \
            --num_objective 2 \
            --eval_way $eval_way\
            --pref_perturb_theta $pref_perturb_theta \
            --weight_num $weight_num \
            --policy_regularization $policy_regularization\
            --eval_freq $eval_freq\
            --pref_gen_way $pref_gen_way\
            > logs/${env}_${dataset_type}_${seed}.txt &
        seed=$(($seed+1))
        gpu=$(($gpu+1))

        policy_regularization='CVAE-QL'
        weight_num=1
        pref_perturb_theta=1.00
        nohup python train_Mujoco_MOOF_TD3_HER.py \
            --env $env --dataset_type $dataset_type \
            --seed $seed \
            --dataset 'd4rl' \
            --gpu $gpu \
            --use_wandb $use_wandb \
            --weight_bc_reward $weight_bc_reward \
            --w_bc_min $w_bc_min \
            --policy_freq $policy_freq \
            --num_objective 2 \
            --eval_way $eval_way\
            --pref_perturb_theta $pref_perturb_theta \
            --weight_num $weight_num \
            --policy_regularization $policy_regularization\
            --eval_freq $eval_freq\
            --pref_gen_way $pref_gen_way\
            > logs/${env}_${dataset_type}_${seed}.txt &
        seed=$(($seed+1))
        gpu=$(($gpu+1))

    done
done    