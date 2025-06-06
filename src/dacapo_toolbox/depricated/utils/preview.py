import matplotlib.pyplot as plt


def batch_to_gif(batch):
    batch_size = len(batch["raw"])
    fig, axes = plt.subplots(batch_size, 3, figsize=(18, 18))
    ims = []
    for zz in range(z_slices):
        b_ims = []
        for bb in range(batch_size):
            b_raw = batch["raw"][bb, 0, zz].numpy()
            b_labels = batch["gt"][bb, zz].numpy() % 256
            b_target = batch["target"][bb, [0, 5, 6], zz].numpy()
            if zz == 0:
                im = axes[bb, 0].imshow(b_raw)
                im2 = axes[bb, 1].imshow(
                    b_labels, cmap=label_cmap, vmin=0, vmax=255, interpolation="none"
                )
                im3 = axes[bb, 2].imshow(
                    b_target.transpose(1, 2, 0), interpolation="none"
                )
                if bb == 0:
                    axes[bb, 0].set_title("Sample Raw")
                    axes[bb, 1].set_title("Sample Labels")
                    axes[bb, 2].set_title("Sample Affinities")
            else:
                im = axes[bb, 0].imshow(b_raw, animated=True)
                im2 = axes[bb, 1].imshow(
                    b_labels,
                    cmap=label_cmap,
                    vmin=0,
                    vmax=255,
                    animated=True,
                    interpolation="none",
                )
                im3 = axes[bb, 2].imshow(
                    b_target.transpose(1, 2, 0), animated=True, interpolation="none"
                )
            b_ims.extend([im, im2, im3])
        ims.append(b_ims)

    ims = ims + ims[::-1]
    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)
    ani.save("_static/minimal_tutorial/affs-batch.gif", writer="pillow", fps=10)
    plt.close()
