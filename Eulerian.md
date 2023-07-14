Eulerian video motion magnification works by analyzing the temporal changes in a video sequence. It does this by taking a standard video sequence as input, and applying [[spatial decomposition]], followed by [[temporal filtering]] to the frames. The resulting signal is then amplified to reveal hidden information.

Eulerian video motion magnification in a nutshell:
1. The first step is to decompose the video sequence into different spatial frequency bands. This is done using a Fourier transform, which decomposes the signal into its constituent frequencies. The different spatial frequency bands represent different levels of detail in the image. For example, the low-frequency bands represent the overall brightness of the image, while the high-frequency bands represent the fine details.
2. The next step is to magnify each spatial frequency band differently. This is done because different spatial frequency bands may contain different amounts of information. For example, the low-frequency bands may contain more noise than the high-frequency bands. As a result, the low-frequency bands may need to be magnified less than the high-frequency bands in order to avoid amplifying the noise
3. The final step is to recombine the amplified spatial frequency bands to form a new video sequence. This new video sequence shows the motion of the object or scene being filmed in more detail.
![[Pasted image 20230713203113.png]]
***Figure 1** : Overview of the Eulerian video magnification framework. The system first decomposes the input video sequence into different spatial frequency bands, and applies the same temporal filter to all bands. The filtered spatial bands are then amplified by a given factor α, added back to the original signal, and collapsed to generate the output video. The choice of temporal filter and amplification factors can be tuned to support different applications. For example, we use the system to reveal unseen motions of a Digital SLR camera, caused by the flipping mirror during a photo burst (camera; full sequences are available in the supplemental video).*

![[Pasted image 20230713232542.png]]
***Figure 2**: Eulerian video magnification used to amplify subtle motions of blood vessels arising from blood flow. For this video, we tuned the temporal filter to a frequency band that includes the heart rate—0.88 Hz (53 bpm)—and set the amplification factor to α = 10. To reduce motion magnification of irrelevant objects, we applied a user-given mask to amplify the area near the wrist only. Movement of the radial and ulnar arteries can barely be seen in the input video (a) taken with a standard point-and-shoot camera, but is significantly more noticeable in the motion-magnified output (b). The motion of the pulsing arteries is more visible when observing a spatio-temporal Y T slice of the wrist (a) and (b).*
