// Direct V4L2 diagnostics: run only while the viewer is stopped.
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <cerrno>
#include <cstring>

static void check(int result, const char* name) {
    if (result < 0) { perror(name); std::exit(1); }
}
int main(int argc, char** argv) {
    if(argc != 5) { std::cerr << "device seconds buffers evidence-prefix\n"; return 2; }
    int fd=open(argv[1],O_RDWR|O_NONBLOCK); check(fd,"open");
    v4l2_format fmt{}; fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width=1344; fmt.fmt.pix.height=376; fmt.fmt.pix.pixelformat=V4L2_PIX_FMT_YUYV;
    check(ioctl(fd,VIDIOC_S_FMT,&fmt),"S_FMT");
    v4l2_streamparm parm{}; parm.type=fmt.type;
    parm.parm.capture.timeperframe.numerator=1; parm.parm.capture.timeperframe.denominator=15;
    check(ioctl(fd,VIDIOC_S_PARM,&parm),"S_PARM");
    v4l2_requestbuffers req{}; req.count=std::atoi(argv[3]); req.type=fmt.type; req.memory=V4L2_MEMORY_MMAP;
    check(ioctl(fd,VIDIOC_REQBUFS,&req),"REQBUFS");
    std::vector<void*> buffers; std::vector<size_t> sizes;
    for(unsigned i=0;i<req.count;++i){
        v4l2_buffer b{}; b.type=fmt.type; b.memory=req.memory; b.index=i;
        check(ioctl(fd,VIDIOC_QUERYBUF,&b),"QUERYBUF");
        void* p=mmap(nullptr,b.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,b.m.offset);
        if(p==MAP_FAILED) {perror("mmap");return 1;}
        buffers.push_back(p); sizes.push_back(b.length); check(ioctl(fd,VIDIOC_QBUF,&b),"QBUF");
    }
    check(ioctl(fd,VIDIOC_STREAMON,&fmt.type),"STREAMON");
    std::cerr << argv[1] << " " << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height << " size=" << fmt.fmt.pix.sizeimage << " buffers=" << req.count << "\n";
    auto start=std::chrono::steady_clock::now();
    unsigned count=0, errors=0, shortframes=0, gaps=0, prev=0, saved=0, greenframes=0;
    std::cout<<"frame,sequence,bytesused,flags,green_row_fraction\n";
    while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<std::atof(argv[2])){
        pollfd pfd{fd,POLLIN,0}; int ret=poll(&pfd,1,2000); check(ret,"poll"); if(!ret)continue;
        v4l2_buffer b{}; b.type=fmt.type; b.memory=req.memory;
        ret=ioctl(fd,VIDIOC_DQBUF,&b); if(ret<0 && errno==EAGAIN)continue; check(ret,"DQBUF");
        ++count; if(count>1 && b.sequence>prev+1)gaps+=b.sequence-prev-1; prev=b.sequence;
        bool error=b.flags & V4L2_BUF_FLAG_ERROR, shortframe=b.bytesused!=fmt.fmt.pix.sizeimage;
        errors+=error; shortframes+=shortframe;
        cv::Mat raw(fmt.fmt.pix.height,fmt.fmt.pix.width,CV_8UC2,buffers.at(b.index),fmt.fmt.pix.bytesperline), rgb;
        cv::cvtColor(raw,rgb,cv::COLOR_YUV2BGR_YUYV);
        double maxgreen=0;
        for(int y=0;y<rgb.rows;++y){unsigned green=0; for(int x=0;x<rgb.cols;++x){auto c=rgb.at<cv::Vec3b>(y,x); green+=(c[1]>70 && c[1]>c[0]+45 && c[1]>c[2]+45);} maxgreen=std::max(maxgreen,double(green)/rgb.cols);}
        greenframes+=maxgreen>.3;
        if(saved<8 && (error||shortframe||maxgreen>.3||count==5)){
            cv::imwrite(std::string(argv[4])+"-"+std::to_string(count)+".jpg",rgb); ++saved;
        }
        std::cout<<count<<","<<b.sequence<<","<<b.bytesused<<","<<b.flags<<","<<maxgreen<<"\n";
        check(ioctl(fd,VIDIOC_QBUF,&b),"QBUF");
    }
    ioctl(fd,VIDIOC_STREAMOFF,&fmt.type);
    for(size_t i=0;i<buffers.size();++i)munmap(buffers[i],sizes[i]);
    close(fd);
    std::cerr<<"frames="<<count<<" flags_error="<<errors<<" wrong_size="<<shortframes<<" sequence_gaps="<<gaps<<" green_frames="<<greenframes<<"\n";
}
