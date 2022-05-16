from model import LayoutGAN


def main():
    layout_gan = LayoutGAN()
    # layout_gan.train()
    layout_gan.test("./checkpoint/state_064.pth")


if __name__ == '__main__':
    main()
