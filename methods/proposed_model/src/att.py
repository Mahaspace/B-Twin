def att(weights, p_target, p_donor_pool, y_target, y_donor_pool):
    target_probability = p_target.mean().detach()
    donor_pool_probability = (p_donor_pool).mean().detach()

    w_target = p_target.unsqueeze(1)
    w_donor_pool = (p_donor_pool).unsqueeze(1)


    donor_pool_weighted_score = weights @ (y_donor_pool * w_donor_pool)

    target_weighted_score = y_target * w_target

    objective = (
        (target_weighted_score / target_probability)
        - (donor_pool_weighted_score / donor_pool_probability)
    ).mean()

    return objective
