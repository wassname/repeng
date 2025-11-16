# repeng (research branch)

This is my rewrite of repeng to
- use hooks not wrapper classes (easier to hack)
- tensors not numpy

see the original [repo](https://github.com/vgel/repeng) for usage. Or my notebooks/try_steering_different_layers_types.ipynb 

```py
with steer(model, vector, coeff):
    out_ids, logr = generate_and_classify(model, input_ids, generation_config, choice_ids)
```