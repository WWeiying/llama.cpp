// Only a trusted OS/platform provider may call this RV64 scalar state reader.
    .text
    .option push
    .option norvc
    .option norelax
    .p2align 2
    .globl akv_resident_platform_read_state_v1
    .type akv_resident_platform_read_state_v1,@function
akv_resident_platform_read_state_v1:
    csrr t0, vstart
    bnez t0, .Lbad_vstart
    csrr t1, vlenb
    csrr t2, fcsr
    csrr t3, vcsr
    sd t1, 0(a1)
    sd t0, 8(a1)
    sd t2, 16(a1)
    sd t3, 24(a1)
    li a0, 1
    ret
.Lbad_vstart:
    li a0, -1
    ret
    .size akv_resident_platform_read_state_v1, .-akv_resident_platform_read_state_v1
    .option pop
    .section .note.GNU-stack,"",@progbits
