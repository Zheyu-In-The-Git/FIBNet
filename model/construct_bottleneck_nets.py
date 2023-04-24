from model import BottleneckNets, ResNet18Encoder,ResNet50Encoder, ResNet101Encoder, \
    UtilityDiscriminator, ResNetDecoder
from model import LitEncoder1, LatentDiscriminator

def ConstructBottleneckNets(args, **kwargs):

    if args.encoder_model == 'ResNet50':
        # 编码器
        get_encoder = ResNet50Encoder(latent_dim=args.latent_dim, channels=3)

        # 解码器
        get_decoder = ResNetDecoder(latent_dim=args.latent_dim, identity_nums=args.identity_nums, s=args.s, m=args.m, easy_margin=False)

        # z判别器
        get_latent_discriminator = LatentDiscriminator(latent_dim=args.latent_dim)

        # u判别器
        get_utility_discriminator = UtilityDiscriminator(utility_dim=args.identity_nums)


    elif args.encoder_model == 'ResNet101':
        # 编码器
        get_encoder = ResNet101Encoder(latent_dim=args.latent_dim, channels=3)

        # 解码器
        get_decoder = ResNetDecoder(latent_dim=args.latent_dim, identity_nums=args.identity_nums, s=args.s, m=args.m, easy_margin=False)

        # z判别器
        get_latent_discriminator = LatentDiscriminator(latent_dim=args.latent_dim)

        # u判别器
        get_utility_discriminator = UtilityDiscriminator(utility_dim=args.identity_nums)


    elif args.encoder_model == 'ResNet18':
        # 编码器
        get_encoder = ResNet18Encoder(latent_dim=args.latent_dim, channels=3)

        # 解码器
        get_decoder = ResNetDecoder(latent_dim=args.latent_dim, identity_nums=args.identity_nums, s=args.s, m=args.m, easy_margin=False)

        # z判别器
        get_latent_discriminator = LatentDiscriminator(latent_dim=args.latent_dim)

        # u判别器
        get_utility_discriminator = UtilityDiscriminator(utility_dim=args.identity_nums)

    elif args.encoder_model == 'LitModel':
        # 编码器
        get_encoder = LitEncoder1(latent_dim=args.latent_dim, channels=3)

        # 解码器
        get_decoder = ResNetDecoder(latent_dim=args.latent_dim, identity_nums=args.identity_nums, s=args.s, m=args.m, easy_margin=False)

        # u判别器
        get_utility_discriminator = UtilityDiscriminator(utility_dim=args.identity_nums)


    return BottleneckNets(model_name= args.model_name,
                          encoder=get_encoder,
                          decoder=get_decoder,
                          utility_discriminator=get_utility_discriminator,
                          latent_discriminator=get_latent_discriminator,
                          batch_size=args.batch_size,
                          beta = args.beta,
                          lr = args.lr,
                          identity_nums = args.identity_nums, **kwargs)



