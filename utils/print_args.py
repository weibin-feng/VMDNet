def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        print(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        print(f'  {"Inverse:":<20}{args.inverse:<20}')
        print()

    if args.task_name == 'imputation':
        print("\033[1m" + "Imputation Task" + "\033[0m")
        print(f'  {"Mask Rate:":<20}{args.mask_rate:<20}')
        print()

    if args.task_name == 'anomaly_detection':
        print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        print(f'  {"Anomaly Ratio:":<20}{args.anomaly_ratio:<20}')
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Top k:":<20}{args.top_k:<20}{"Num Kernels:":<20}{args.num_kernels:<20}')
    print(f'  {"Enc In:":<20}{args.enc_in:<20}{"Dec In:":<20}{args.dec_in:<20}')
    print(f'  {"C Out:":<20}{args.c_out:<20}{"d model:":<20}{args.d_model:<20}')
    print(f'  {"n heads:":<20}{args.n_heads:<20}{"e layers:":<20}{args.e_layers:<20}')
    print(f'  {"d layers:":<20}{args.d_layers:<20}{"d FF:":<20}{args.d_ff:<20}')
    print(f'  {"Moving Avg:":<20}{args.moving_avg:<20}{"Factor:":<20}{args.factor:<20}')
    print(f'  {"Distil:":<20}{args.distil:<20}{"Dropout:":<20}{args.dropout:<20}')
    print(f'  {"Embed:":<20}{args.embed:<20}{"Activation:":<20}{args.activation:<20}')
    print(f'  {"Channel Indep:":<20}{args.channel_independence:<20}{"Decomp Method:":<20}{args.decomp_method:<20}')
    print(f'  {"Use Norm:":<20}{args.use_norm:<20}{"Down Samp Layers:":<20}{args.down_sampling_layers:<20}')
    # print(f'  {"Down Samp Window:":<20}{args.down_sampling_window:<20}'
    #       f'{"Down Samp Method:":<20}{str(args.down_sampling_method):<20}')  # <--- 修改此处

    print(f'  {"Seg Len:":<20}{args.seg_len:<20}{"Patch Len:":<20}{args.patch_len:<20}')
    print()

    # --- ADDED: VMD Parameters ---
    print("\033[1m" + "VMD Parameters" + "\033[0m")
    print(f'  {"VMD K:":<20}{args.vmd_K:<20}{"VMD Alpha:":<20}{args.vmd_alpha:<20}')
    print(f'  {"VMD Tau:":<20}{args.vmd_tau:<20}{"VMD Tol:":<20}{args.vmd_tol:<20}')
    print()

    # --- ADDED: TCN Decoder Parameters ---
    print("\033[1m" + "TCN Decoder Parameters" + "\033[0m")
    tcn_hdims_str = ', '.join(map(str, args.tcn_hidden_dims)) # Use the combined list
    print(f'  {"TCN Hidden Dims:":<20}{tcn_hdims_str:<20}{"TCN Dropout:":<20}{args.tcn_dropout:<20}')
    print()

    # --- ADDED: Fusion Parameters ---
    print("\033[1m" + "Fusion Parameters" + "\033[0m")
    print(f'  {"Fusion Method:":<20}{args.fusion_method:<20}', end='')
    if args.fusion_method == 'ffn': # Only print ffn_hidden_dim if method is ffn
        print(f'{"FFN Hidden Dim:":<20}{args.ffn_hidden_dim:<20}')
    else:
        print() # Print a newline if not ffn to maintain formatting
    print()


    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    print(f'  {"Lradj:":<20}{args.lradj:<20}{"Use Amp:":<20}{args.use_amp:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU Type:":<20}{args.gpu_type:<20}') # Added GPU Type
    print(f'  {"GPU ID:":<20}{args.gpu:<20}{"Use Multi GPU:":<20}{args.use_multi_gpu:<20}') # Changed "GPU:" to "GPU ID:"
    print(f'  {"Devices:":<20}{args.devices:<20}') # Devices might be long, keep it on its own line for readability
    print()


    print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    p_hidden_dims_str = ', '.join(map(str, args.p_hidden_dims))
    print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{args.p_hidden_layers:<20}')
    print()

    print("\033[1m" + "Metrics" + "\033[0m") # Added new section for metrics
    print(f'  {"Use DTW:":<20}{args.use_dtw:<20}')
    print()

    # --- ADDED: Augmentation Parameters ---
    if any([args.augmentation_ratio, args.jitter, args.scaling, args.permutation, args.randompermutation,
            args.magwarp, args.timewarp, args.windowslice, args.windowwarp, args.rotation,
            args.spawner, args.dtwwarp, args.shapedtwwarp, args.wdba, args.discdtw, args.discsdtw]):
        print("\033[1m" + "Augmentation Parameters" + "\033[0m")
        print(f'  {"Augmentation Ratio:":<20}{args.augmentation_ratio:<20}{"Seed:":<20}{args.seed:<20}')
        print(f'  {"Jitter:":<20}{args.jitter:<20}{"Scaling:":<20}{args.scaling:<20}')
        print(f'  {"Permutation:":<20}{args.permutation:<20}{"Random Permutation:":<20}{args.randompermutation:<20}')
        print(f'  {"Mag Warp:":<20}{args.magwarp:<20}{"Time Warp:":<20}{args.timewarp:<20}')
        print(f'  {"Window Slice:":<20}{args.windowslice:<20}{"Window Warp:":<20}{args.windowwarp:<20}')
        print(f'  {"Rotation:":<20}{args.rotation:<20}{"Spawner:":<20}{args.spawner:<20}')
        print(f'  {"DTW Warp:":<20}{args.dtwwarp:<20}{"Shape DTW Warp:":<20}{args.shapedtwwarp:<20}')
        print(f'  {"Weighted DBA:":<20}{args.wdba:<20}{"Disc DTW:":<20}{args.discdtw:<20}')
        print(f'  {"Disc Shape DTW:":<20}{args.discsdtw:<20}{"Extra Tag:":<20}{args.extra_tag:<20}')
        print()