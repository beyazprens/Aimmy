using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using Other;

namespace MouseMovementLibraries.VolmgrSupport
{
    internal class VolmgrMouse
    {
        private const string DEVICE_NAME = @"\\.\volmgra";
        private const uint IO_SEND_MOUSE_EVENT = 0x1FE3BD28;

        private static SafeFileHandle? _deviceHandle;
        private static readonly object _lock = new();

        [StructLayout(LayoutKind.Sequential)]
        private struct NF_MOUSE_REQUEST
        {
            public int X;
            public int Y;
            public short ButtonFlags;
        }

        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern SafeFileHandle CreateFile(
            string lpFileName,
            uint dwDesiredAccess,
            uint dwShareMode,
            IntPtr lpSecurityAttributes,
            uint dwCreationDisposition,
            uint dwFlagsAndAttributes,
            IntPtr hTemplateFile);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool DeviceIoControl(
            SafeFileHandle hDevice,
            uint dwIoControlCode,
            ref NF_MOUSE_REQUEST lpInBuffer,
            int nInBufferSize,
            IntPtr lpOutBuffer,
            int nOutBufferSize,
            out int lpBytesReturned,
            IntPtr lpOverlapped);

        public static bool Load()
        {
            try
            {
                lock (_lock)
                {
                    if (_deviceHandle != null && !_deviceHandle.IsInvalid && !_deviceHandle.IsClosed)
                        return true;

                    _deviceHandle = CreateFile(
                        DEVICE_NAME,
                        0xC0000000, // GENERIC_READ | GENERIC_WRITE
                        0,
                        IntPtr.Zero,
                        3, // OPEN_EXISTING
                        0,
                        IntPtr.Zero);

                    if (_deviceHandle.IsInvalid)
                    {
                        LogManager.Log(LogManager.LogLevel.Error, "Volmgr driver connection failed. Make sure the driver is loaded.", true);
                        return false;
                    }

                    LogManager.Log(LogManager.LogLevel.Info, "Volmgr driver loaded successfully.", true);
                    return true;
                }
            }
            catch (Exception ex)
            {
                LogManager.Log(LogManager.LogLevel.Error, $"Volmgr driver load error: {ex.Message}", true);
                return false;
            }
        }

        public static void Move(int dx, int dy)
        {
            if (dx == 0 && dy == 0) return;

            lock (_lock)
            {
                if (_deviceHandle == null || _deviceHandle.IsInvalid || _deviceHandle.IsClosed)
                    return;

                var request = new NF_MOUSE_REQUEST
                {
                    X = dx,
                    Y = dy,
                    ButtonFlags = 0
                };

                DeviceIoControl(
                    _deviceHandle,
                    IO_SEND_MOUSE_EVENT,
                    ref request,
                    Marshal.SizeOf<NF_MOUSE_REQUEST>(),
                    IntPtr.Zero,
                    0,
                    out _,
                    IntPtr.Zero);
            }
        }

        public static void Click(short buttonFlags)
        {
            lock (_lock)
            {
                if (_deviceHandle == null || _deviceHandle.IsInvalid || _deviceHandle.IsClosed)
                    return;

                var request = new NF_MOUSE_REQUEST
                {
                    X = 0,
                    Y = 0,
                    ButtonFlags = buttonFlags
                };

                DeviceIoControl(
                    _deviceHandle,
                    IO_SEND_MOUSE_EVENT,
                    ref request,
                    Marshal.SizeOf<NF_MOUSE_REQUEST>(),
                    IntPtr.Zero,
                    0,
                    out _,
                    IntPtr.Zero);
            }
        }

        public static void Close()
        {
            lock (_lock)
            {
                if (_deviceHandle != null && !_deviceHandle.IsInvalid && !_deviceHandle.IsClosed)
                {
                    _deviceHandle.Close();
                    _deviceHandle = null;
                }
            }
        }
    }
}
