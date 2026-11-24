//! Emulates CLZERO on CPUs that raise an illegal-instruction exception for it.

use std::ffi::c_void;
use windows_sys::Win32::Foundation::{GetLastError, SetLastError};
use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, ReadProcessMemory};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY,
    PAGE_NOCACHE, PAGE_READWRITE, PAGE_WRITECOMBINE, PAGE_WRITECOPY, VirtualQuery,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const OP: [u8; 3] = [0x0f, 0x01, 0xfc];
const LINE: usize = 64;
const LOOP: usize = 12;
const PAGE: usize = 4096;
const MAX_LOOPS: usize = 5;
const ALIGN_NOP: [u8; 10] = [0x66, 0x2e, 0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0];
const MAX_CODE: usize = LOOP + (MAX_LOOPS - 1) * (LOOP + 10 + ALIGN_NOP.len());
static ZEROES: [u8; 64 * 1024] = [0; 64 * 1024];

type Bounds = [u64; 4];

struct Reset {
    len: usize,
    tail: usize,
    bounds: Bounds,
}

fn bound_slot(register: u8) -> Option<usize> {
    match register {
        2 => Some(0),  // RDX
        8 => Some(1),  // R8
        9 => Some(2),  // R9
        10 => Some(3), // R10
        _ => None,
    }
}

/// CLZERO; ADD RAX, 64; CMP RAX, bound; JB back to CLZERO.
fn loop_bound(code: &[u8]) -> Option<usize> {
    let [
        0x0f,
        0x01,
        0xfc,
        0x48,
        0x83,
        0xc0,
        0x40,
        rex,
        0x39,
        cmp,
        0x72,
        0xf4,
    ] = *code.first_chunk::<LOOP>()?
    else {
        return None;
    };
    if !matches!(rex, 0x48 | 0x4c) || cmp & 0xc7 != 0xc0 {
        return None;
    }
    bound_slot(((cmp >> 3) & 7) | ((rex & 4) << 1))
}

/// MOV RAX, bound; ADD bound, positive imm32; optional alignment NOP.
fn transition(code: &[u8]) -> Option<(usize, u64, usize)> {
    let [rex, 0x89, mov, add_rex, 0x81, add, a, b, c, d] = *code.first_chunk::<10>()? else {
        return None;
    };
    if !matches!(rex, 0x48 | 0x4c) || mov & 0xc7 != 0xc0 {
        return None;
    }
    let register = ((mov >> 3) & 7) | ((rex & 4) << 1);
    if add_rex != (0x48 | (register >> 3)) || add != (0xc0 | (register & 7)) {
        return None;
    }
    let stride = u64::try_from(i32::from_le_bytes([a, b, c, d])).ok()?;
    if stride == 0 || stride % LINE as u64 != 0 {
        return None;
    }
    let padding = if code[10..].starts_with(&ALIGN_NOP) {
        ALIGN_NOP.len()
    } else {
        0
    };
    Some((bound_slot(register)?, stride, 10 + padding))
}

/// Fuse neighbouring loops only when each starts exactly where the last
/// ended. A failed transition leaves the plan for the verified prefix.
fn reset_plan(code: &[u8], start: u64, mut bounds: Bounds) -> Option<Reset> {
    let mut end = bounds[loop_bound(code)?];
    let valid_end = |end: u64| {
        end.checked_sub(start)
            .is_some_and(|len| len != 0 && len % LINE as u64 == 0 && len <= ZEROES.len() as u64)
    };
    if !valid_end(end) {
        return None;
    }
    let mut tail = 0;
    for _ in 1..MAX_LOOPS {
        let Some((slot, stride, advance)) = transition(&code[tail + LOOP..]) else {
            break;
        };
        let next = tail + LOOP + advance;
        let Some(limit) = loop_bound(&code[next..]) else {
            break;
        };
        if bounds[slot] != end {
            break;
        }
        let Some(increased) = end.checked_add(stride) else {
            break;
        };
        let next_end = if slot == limit {
            increased
        } else {
            bounds[limit]
        };
        if next_end <= end || !valid_end(next_end) {
            break;
        }
        bounds[slot] = increased;
        end = next_end;
        tail = next;
    }
    (tail != 0).then_some(Reset {
        len: (end - start) as usize,
        tail,
        bounds,
    })
}

/// The kernel validates both spans without changing their protections.
fn copy(src: *const c_void, dst: *mut c_void, len: usize) -> bool {
    let mut copied = 0;
    // SAFETY: the kernel validates the spans and writes the count to this local.
    let ok = unsafe { ReadProcessMemory(GetCurrentProcess(), src, dst, len, &mut copied) };
    ok != 0 && copied == len
}

pub(crate) fn emulate(context: &mut CONTEXT) -> bool {
    // SAFETY: these calls access the current thread's last-error slot.
    let error = unsafe { GetLastError() };
    let handled = zero(context);
    // SAFETY: restore the error value from before emulation.
    unsafe { SetLastError(error) };
    handled
}

fn zero(context: &mut CONTEXT) -> bool {
    let ip = context.Rip as usize;
    let line = (context.Rax as usize) & !(LINE - 1);
    let mut region = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: VirtualQuery does not dereference line and writes into this local.
    let queried = unsafe { VirtualQuery(line as *const c_void, &mut region, size_of_val(&region)) };
    let start = region.BaseAddress as usize;
    let protection = region.Protect & !(PAGE_NOCACHE | PAGE_WRITECOMBINE);
    let Some(writable) = start
        .checked_add(region.RegionSize)
        .and_then(|end| end.checked_sub(line))
    else {
        return false;
    };
    let batchable = matches!(protection, PAGE_READWRITE | PAGE_WRITECOPY);
    let writable_code = matches!(protection, PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY);
    if queried != size_of_val(&region)
        || region.State != MEM_COMMIT
        || !(batchable || writable_code)
        || line < start
        || writable < LINE
    {
        return false;
    }

    let zero_lines = |len| copy(ZEROES.as_ptr().cast(), line as *mut c_void, len);
    let mut len = LINE;
    // Leave instruction stepping and hardware breakpoints to the debugger.
    if batchable && context.EFlags & 0x100 == 0 && context.Dr7 & 0xff == 0 {
        let mut code = [0; MAX_CODE];
        let n = (PAGE - (ip & (PAGE - 1))).min(code.len());
        if n >= LOOP && copy(ip as *const c_void, code.as_mut_ptr().cast(), n) {
            let bounds = [context.Rdx, context.R8, context.R9, context.R10];
            let code = &code[..n];
            if let Some(reset) = reset_plan(code, context.Rax, bounds)
                && reset.len <= writable
                && zero_lines(reset.len)
            {
                [context.Rdx, context.R8, context.R9, context.R10] = reset.bounds;
                context.Rax += (reset.len - LINE) as u64;
                // Execute the last ADD/CMP/JB natively to set RAX and flags.
                context.Rip += (reset.tail + OP.len()) as u64;
                return true;
            }
            if let Some(slot) = loop_bound(code) {
                let lines = bounds[slot]
                    .saturating_sub(context.Rax)
                    .div_ceil(LINE as u64)
                    .max(1);
                let max_lines = writable.min(ZEROES.len()) / LINE;
                len = lines.min(max_lines as u64) as usize * LINE;
            }
        }
    }
    if !zero_lines(len) {
        // A protection change can invalidate part of a previously checked span.
        len = LINE;
        if !zero_lines(len) {
            return false;
        }
    }
    context.Rax += (len - LINE) as u64;
    context.Rip += OP.len() as u64;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::null;
    use windows_sys::Win32::System::Memory::{
        MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc, VirtualFree,
    };

    const R10_LOOP: [u8; LOOP] = [
        0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x4c, 0x39, 0xd0, 0x72, 0xf4,
    ];
    const RDX_LOOP: [u8; LOOP] = [
        0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x48, 0x39, 0xd0, 0x72, 0xf4,
    ];
    // mov rax, r10; add r10, 2 lines
    const NEXT_R10: [u8; 10] = [0x4c, 0x89, 0xd0, 0x49, 0x81, 0xc2, 0x80, 0, 0, 0];
    // mov rax, r9; add r9, 3 lines
    const NEXT_R9: [u8; 10] = [0x4c, 0x89, 0xc8, 0x49, 0x81, 0xc1, 0xc0, 0, 0, 0];

    #[repr(align(4096))]
    struct Aligned([u8; PAGE]);

    fn fixture() -> (Aligned, usize) {
        let mut code = Aligned([0; PAGE]);
        let loops = [
            &R10_LOOP[..],
            &NEXT_R10,
            &ALIGN_NOP,
            &R10_LOOP,
            &NEXT_R9,
            &RDX_LOOP,
        ]
        .concat();
        code.0[..loops.len()].copy_from_slice(&loops);
        (code, loops.len())
    }

    fn registers(ip: u64, base: u64) -> CONTEXT {
        CONTEXT {
            Rip: ip,
            Rax: base,
            R10: base + 2 * LINE as u64,
            R9: base + 4 * LINE as u64,
            Rdx: base + 6 * LINE as u64,
            EFlags: 0x202,
            ..Default::default()
        }
    }

    #[test]
    fn combines_adjacent_loops_and_leaves_the_last_comparison_native() {
        let (code, n) = fixture();
        let mut data = Aligned([0xa5; PAGE]);
        let base = data.0.as_mut_ptr() as u64;
        let ip = code.0.as_ptr() as u64;
        let mut cpu = registers(ip, base);
        let original_flags = cpu.EFlags;

        assert!(emulate(&mut cpu));
        assert_eq!(cpu.Rip, ip + (n - LOOP + OP.len()) as u64);
        assert_eq!(cpu.Rax, base + 5 * LINE as u64);
        assert_eq!(cpu.R10, base + 4 * LINE as u64);
        assert_eq!(cpu.R9, base + 7 * LINE as u64);
        assert_eq!(cpu.Rdx, base + 6 * LINE as u64);
        assert_eq!(cpu.EFlags, original_flags);
        assert!(data.0[..6 * LINE].iter().all(|&byte| byte == 0));
        assert!(data.0[6 * LINE..].iter().all(|&byte| byte == 0xa5));
    }

    #[test]
    fn stops_before_gaps_and_rejects_a_changed_branch() {
        let (mut code, n) = fixture();
        let bounds = [384, 0, 320, 128];
        let reset = reset_plan(&code.0[..n], 0, bounds).unwrap();
        assert_eq!(reset.len, 4 * LINE);
        assert_eq!(reset.bounds[2], bounds[2]);

        code.0[LOOP + NEXT_R10.len() + ALIGN_NOP.len() + LOOP - 1] ^= 1;
        assert!(reset_plan(&code.0[..n], 0, bounds).is_none());
    }

    #[test]
    fn stepping_emulates_only_the_current_line() {
        let (code, _) = fixture();
        let mut data = Aligned([0xa5; PAGE]);
        let base = data.0.as_mut_ptr() as u64;
        let ip = code.0.as_ptr() as u64;
        let mut cpu = registers(ip, base);
        cpu.EFlags |= 0x100;

        assert!(emulate(&mut cpu));
        assert_eq!(cpu.Rip, ip + OP.len() as u64);
        assert_eq!(cpu.Rax, base);
        assert!(data.0[..LINE].iter().all(|&byte| byte == 0));
        assert!(data.0[LINE..].iter().all(|&byte| byte == 0xa5));
    }

    #[test]
    fn does_not_batch_past_the_end_of_a_writable_region() {
        let (code, _) = fixture();
        // SAFETY: reserve two private pages and commit only the first one.
        let pages = unsafe { VirtualAlloc(null(), 2 * PAGE, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!pages.is_null());
        let committed = unsafe { VirtualAlloc(pages, PAGE, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(committed, pages);

        let base = pages as usize + PAGE - 3 * LINE;
        let ptr = base as *mut u8;
        // SAFETY: this span is within the first committed page.
        unsafe { ptr.write_bytes(0xa5, 3 * LINE) };
        let ip = code.0.as_ptr() as u64;
        let mut cpu = registers(ip, base as u64);
        let handled = emulate(&mut cpu);
        // SAFETY: read the same committed span and free the original reservation.
        let observed = unsafe { ptr.cast::<[u8; 3 * LINE]>().read() };
        assert_ne!(unsafe { VirtualFree(pages, 0, MEM_RELEASE) }, 0);

        assert!(handled);
        assert_eq!(cpu.Rip, ip + OP.len() as u64);
        assert_eq!(cpu.Rax, base as u64 + LINE as u64);
        assert_eq!(cpu.R10, base as u64 + 2 * LINE as u64);
        assert_eq!(observed[..2 * LINE], [0; 2 * LINE]);
        assert_eq!(observed[2 * LINE..], [0xa5; LINE]);
    }
}
