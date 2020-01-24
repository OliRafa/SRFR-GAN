  elif args.loss_type==4:
    # S Value
    s = args.margin_s

    # Margin value
    m = args.margin_m

    # Asserting S is positive
    assert s>0.0

    # Asserting Margin is positive and more than 90 degrees
    assert m>=0.0
    assert m<(math.pi/2)

    # Normalizing Weights
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')

    # Normalizing Embeddings
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s

    # Cos Thetaj (Mult. between Norm Weights and Norm Embeddings)
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes, name='fc7')
    
    # Original_target_logit (??)
    zy = mx.sym.pick(fc7, gt_label, axis=1)

    # Arccos
    cos_t = zy/s
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = math.sin(math.pi-m)*m
    #threshold = 0.0
    threshold = math.cos(math.pi-m)
    if args.easy_margin:
      cond = mx.symbol.Activation(data=cos_t, act_type='relu')
    else:
      cond_v = cos_t - threshold
      cond = mx.symbol.Activation(data=cond_v, act_type='relu')
    
    
    body = cos_t*cos_t
    body = 1.0-body
    sin_t = mx.sym.sqrt(body)

    # Cos(THETAyi + Margin) = Cos THETAj
    new_zy = cos_t*cos_m
    b = sin_t*sin_m
    new_zy = new_zy - b

    # Cos THETAj * s --- Feature Rescale
    new_zy = new_zy*s
    if args.easy_margin:
      zy_keep = zy
    else:
      zy_keep = zy - s*mm
    new_zy = mx.sym.where(cond, new_zy, zy_keep)

    diff = new_zy - zy
    diff = mx.sym.expand_dims(diff, 1)

    # Creating One-hot vector
    gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes, on_value = 1.0, off_value = 0.0)
    body = mx.sym.broadcast_mul(gt_one_hot, diff)
    fc7 = fc7+body