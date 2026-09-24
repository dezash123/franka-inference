/* Inject V4L2 delivery results without opening a camera. */
#include <sys/ioctl.h>
#include <poll.h>
#include <stdarg.h>
#include <assert.h>
#include <stdio.h>
static int mock_ioctl(int, unsigned long, ...);
static int mock_poll(struct pollfd *, nfds_t, int);
#define ioctl mock_ioctl
#define poll mock_poll
#include "../capture_v4l2.c"
#undef ioctl
#undef poll
static struct v4l2_buffer delivered;
static int queued;
static int mock_poll(struct pollfd *p, nfds_t n, int timeout) {
    (void)n; (void)timeout; p->revents=POLLIN; return 1;
}
static int mock_ioctl(int fd, unsigned long op, ...) {
    (void)fd;
    va_list args; va_start(args,op); void *arg=va_arg(args,void *); va_end(args);
    if(op==VIDIOC_DQBUF) {*(struct v4l2_buffer *)arg=delivered;return 0;}
    if(op==VIDIOC_QBUF) {queued++;return 0;}
    errno=EINVAL;return -1;
}
int main(void) {
    unsigned char input[12]={1,2,3,4,99,99,5,6,7,8,99,99}, output[8];
    struct capture c={.fd=1,.width=2,.height=2,.stride=6,.size=12,.count=1};
    c.buffers[0]=input; c.lengths[0]=sizeof(input);
    struct frame_meta meta={0};
    delivered.bytesused=12;
    assert(capture_read(&c,output,sizeof(output),&meta)==1);
    const unsigned char expected[8]={1,2,3,4,5,6,7,8};
    assert(memcmp(output,expected,sizeof(output))==0);
    for(int scenario=0;scenario<4;scenario++) {
        delivered.flags=scenario==0?V4L2_BUF_FLAG_ERROR:0;
        delivered.bytesused=scenario==1?6:scenario==2?13:12;
        delivered.index=scenario==3?1:0;
        memset(output,42,sizeof(output));
        assert(capture_read(&c,output,sizeof(output),&meta)==2);
        for(size_t i=0;i<sizeof(output);i++) assert(output[i]==42);
    }
    assert(queued==5);
    assert(capture_read(&c,output,1,&meta)==-1 && errno==ENOSPC);
    puts("PASS: padded valid frame, full-size error, short/oversized payload, invalid index, output bounds; rejected frames never copied.");
}
