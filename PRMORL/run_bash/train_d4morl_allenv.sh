for dataset_type in amateur_uniform expert_uniform
do
    use_wandb=1
    w_bc_min=0.20    
    policy_freq=2           
    eval_way="mc|exhaust" 
    eval_freq=1000000 
    time_steps=1000000 #temp
    pref_gen_way='L1_return'

    for env in Walker2d Swimmer Ant HalfCheetah Hopper     
    do
        seed=2207
        mc_update_num=20
        lr_decay=1
        if [ "$dataset_type" = "expert_uniform" ]; then
            pref_perturb_theta=0.0
        else
            pref_perturb_theta=1.0
        fi

        if [ "$env" = "Hopper" ]; then
            gpu=1
            weight_bc_reward=100.0
            diffusion_n_timesteps=5
        elif [ "$env" = "Ant" ]; then
            gpu=3
            weight_bc_reward=100.0
            diffusion_n_timesteps=5
        elif [ "$env" = "Swimmer" ]; then
            gpu=4
            weight_bc_reward=100.0
            diffusion_n_timesteps=5
        elif [ "$env" = "Walker2d" ]; then
            gpu=5
            weight_bc_reward=60.0
            diffusion_n_timesteps=5
            pref_perturb_theta=0.0
            mc_update_num=100
            lr_decay=0
        elif [ "$env" = "HalfCheetah" ]; then
            gpu=6
            weight_bc_reward=100.0
            diffusion_n_timesteps=5
        fi

       
        

        policy_regularization='Diffusion-QL'
        weight_num=1
        normalize_states=1
        nohup python train_Mujoco_MOOF_TD3_HER.py \
            --env $env --dataset_type $dataset_type \
            --seed $seed \
            --dataset 'd4morl' \
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
            --normalize_states $normalize_states\
            --lr_decay $lr_decay\
            --mc_update_num $mc_update_num\
            > logs/${env}_${dataset_type}_${seed}.txt &
        seed=$(($seed+1))
        gpu=$(($gpu+1))


        policy_regularization='TD3+BC'
        weight_num=1
        normalize_states=1
        pref_perturb_theta=1.00
        weight_bc_reward=50.0
        nohup python train_Mujoco_MOOF_TD3_HER.py \
            --env $env --dataset_type $dataset_type \
            --seed $seed \
            --dataset 'd4morl' \
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
            --normalize_states $normalize_states\
            > logs/${env}_${dataset_type}_${seed}.txt &
        seed=$(($seed+1))
        gpu=$(($gpu+1))


        policy_regularization='CVAE-QL'
        weight_num=1
        normalize_states=1
        pref_perturb_theta=1.00
        weight_bc_reward=200.0
        nohup python train_Mujoco_MOOF_TD3_HER.py \
            --env $env --dataset_type $dataset_type \
            --seed $seed \
            --dataset 'd4morl' \
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
            --normalize_states $normalize_states\
            > logs/${env}_${dataset_type}_${seed}.txt &
        seed=$(($seed+1))
        gpu=$(($gpu+1))

    done
done    