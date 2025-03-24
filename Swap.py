def swap_n_show(img1_fn,img2_fn,app,swapper,plot_before=True,plot_after=True):
    """
    uses face swapper to swap faces in two different images

    plot_before:if true shows the images before the swap
    plot_after:if true shows the images after swap

    returns image with swapped faces

    assumes one face image
    """

    img1=cv2.imread(img1_fn)
    img2=cv2.imread(img2_fn)

    if plot_before:
        fig,axs=plt.subplots(1,2,figsize=(10,5))
        axs[0].imshow(img1[:,:,::-1])
        axs[0].axis('off')
        axs[1].imshow(img2[:,:,::-1])
        axs[1].axis('off')
        plt.show()

    #do the swap
    face1=app.get(img1)[0]
    face2=app.get(img2)[0]

    img1_=img1.copy()
    img2_=img2.copy()

    if plot_after:
        img1_=swapper.get(img1_,face1,face2,paste_back=True)
        img2_=swapper.get(img2_,face2,face1,paste_back=True)
        fig,axs=plt.subplots(1,2,figsize=(10,5))
        axs[0].imshow(img1_[:,:,::-1])
        axs[0].axis('off')
        axs[1].imshow(img2_[:,:,::-1])
        axs[1].axis('off')
        plt.show()

    return img1_,img2_


