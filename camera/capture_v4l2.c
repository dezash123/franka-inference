/* Small Linux YUYV capture interface. Copies only complete, unflagged frames.
 * No USB controls, kernel parameters, or robot interfaces are modified. */
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct capture {
    int fd, streaming;
    uint32_t width, height, stride, size, count;
    void *buffers[4];
    size_t lengths[4];
};
struct frame_meta { uint32_t sequence, flags, bytesused; uint64_t timestamp_us; };
static int xioctl(int fd, unsigned long op, void *arg) {
    int r; do { r=ioctl(fd,op,arg); } while(r<0 && errno==EINTR); return r;
}
/* Read-only discovery: distinguish D405 color from its depth/IR nodes. */
int capture_has_yuyv(const char *device) {
    int fd=open(device,O_RDONLY|O_NONBLOCK|O_CLOEXEC);
    if(fd<0) return 0;
    struct v4l2_fmtdesc fmt={0};
    fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    int found=0;
    while(xioctl(fd,VIDIOC_ENUM_FMT,&fmt)==0) {
        if(fmt.pixelformat==V4L2_PIX_FMT_YUYV) {found=1;break;}
        fmt.index++;
    }
    close(fd);
    return found;
}
void capture_close(struct capture *c) {
    if(!c) return;
    enum v4l2_buf_type type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if(c->streaming) xioctl(c->fd,VIDIOC_STREAMOFF,&type);
    for(uint32_t i=0;i<c->count;i++) if(c->buffers[i]) munmap(c->buffers[i],c->lengths[i]);
    if(c->fd>=0) close(c->fd);
    free(c);
}
struct capture *capture_open(const char *device, uint32_t width, uint32_t height,
                             uint32_t fps, uint32_t *geometry) {
    struct capture *c=calloc(1,sizeof(*c));
    if(!c) return NULL;
    c->fd=open(device,O_RDWR|O_NONBLOCK|O_CLOEXEC);
    if(c->fd<0) goto fail;
    struct v4l2_format fmt={0}; fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width=width; fmt.fmt.pix.height=height;
    fmt.fmt.pix.pixelformat=V4L2_PIX_FMT_YUYV; fmt.fmt.pix.field=V4L2_FIELD_ANY;
    if(xioctl(c->fd,VIDIOC_S_FMT,&fmt)<0) goto fail;
    c->width=fmt.fmt.pix.width; c->height=fmt.fmt.pix.height;
    c->stride=fmt.fmt.pix.bytesperline; c->size=fmt.fmt.pix.sizeimage;
    if(fmt.fmt.pix.pixelformat!=V4L2_PIX_FMT_YUYV || c->width!=width ||
       c->height!=height || c->stride<width*2 || c->size<c->stride*height) {
        errno=EINVAL; goto fail;
    }
    struct v4l2_streamparm parm={0}; parm.type=fmt.type;
    parm.parm.capture.timeperframe.numerator=1; parm.parm.capture.timeperframe.denominator=fps;
    if(xioctl(c->fd,VIDIOC_S_PARM,&parm)<0) goto fail;
    struct v4l2_requestbuffers req={0}; req.type=fmt.type; req.memory=V4L2_MEMORY_MMAP; req.count=4;
    if(xioctl(c->fd,VIDIOC_REQBUFS,&req)<0) goto fail;
    if(req.count<2 || req.count>4) {errno=ENOMEM; goto fail;}
    c->count=req.count;
    for(uint32_t i=0;i<c->count;i++) {
        struct v4l2_buffer b={0}; b.type=fmt.type; b.memory=req.memory; b.index=i;
        if(xioctl(c->fd,VIDIOC_QUERYBUF,&b)<0) goto fail;
        if(b.length<c->size) {errno=EINVAL; goto fail;}
        void *p=mmap(NULL,b.length,PROT_READ|PROT_WRITE,MAP_SHARED,c->fd,b.m.offset);
        if(p==MAP_FAILED) goto fail;
        c->buffers[i]=p; c->lengths[i]=b.length;
        if(xioctl(c->fd,VIDIOC_QBUF,&b)<0) goto fail;
    }
    if(xioctl(c->fd,VIDIOC_STREAMON,&fmt.type)<0) goto fail;
    c->streaming=1;
    geometry[0]=c->width; geometry[1]=c->height;
    geometry[2]=parm.parm.capture.timeperframe.numerator;
    geometry[3]=parm.parm.capture.timeperframe.denominator;
    return c;
fail:;
    int saved=errno; capture_close(c); errno=saved; return NULL;
}
/* 1=accepted, 2=corrupt/incomplete and discarded, 0=timeout, -1=I/O error. */
int capture_read(struct capture *c, unsigned char *output, size_t capacity,
                 struct frame_meta *meta) {
    if(capacity<(size_t)c->width*c->height*2) {errno=ENOSPC;return -1;}
    struct pollfd p={c->fd,POLLIN,0};
    int r=poll(&p,1,500);
    if(r<0 && errno==EINTR) return 0;
    if(r<=0) return r;
    if(p.revents&(POLLERR|POLLHUP|POLLNVAL)) {errno=ENODEV;return -1;}
    struct v4l2_buffer b={0}; b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory=V4L2_MEMORY_MMAP;
    if(xioctl(c->fd,VIDIOC_DQBUF,&b)<0) return errno==EAGAIN?0:-1;
    meta->sequence=b.sequence; meta->flags=b.flags; meta->bytesused=b.bytesused;
    meta->timestamp_us=(uint64_t)b.timestamp.tv_sec*1000000+b.timestamp.tv_usec;
    int valid=b.index<c->count && !(b.flags&V4L2_BUF_FLAG_ERROR) && b.bytesused==c->size;
    if(valid) {
        for(uint32_t y=0;y<c->height;y++)
            memcpy(output+(size_t)y*c->width*2,(unsigned char*)c->buffers[b.index]+(size_t)y*c->stride,c->width*2);
    }
    if(xioctl(c->fd,VIDIOC_QBUF,&b)<0) return -1;
    return valid?1:2;
}
