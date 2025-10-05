using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text;
using System.Threading.Tasks;

namespace Alturos.Yolo
{
    /// <summary>
    /// Helper function to dynamically load DLL contained functions on Windows only
    /// </summary>
    internal class FunctionLoader
    {
        [DllImport("Kernel32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
        private static extern IntPtr LoadLibrary(string path);

        [DllImport("Kernel32.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr GetProcAddress(IntPtr hModule, string procName);

        /// <summary>
        /// Map String (library name) to IntPtr (reference from LoadLibrary)
        /// </summary>
        private static ConcurrentDictionary<string, IntPtr> LoadedLibraries { get; } = new ConcurrentDictionary<string, IntPtr>();

        /// <summary>
        /// Load function (by name) from DLL (by name) and return its delegate
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="dllPath"></param>
        /// <param name="functionName"></param>
        /// <returns></returns>
        public static T LoadFunction<T>(string dllPath, string functionName)
        {
            // normalize
            if (!File.Exists(dllPath)) 
            {
                dllPath = Path.GetFullPath(dllPath);
            }


            // Get preloaded or load the library on-demand
            IntPtr hModule = LoadedLibraries.GetOrAdd(
                dllPath,
                valueFactory: (string dp) =>
                {
                    IntPtr loaded = LoadLibrary(dp);
                    if (loaded == IntPtr.Zero)
                    {
                        throw new DllNotFoundException($"Library missing at path {dp}");
                    }
                    return loaded;
                }
            );
            // Load function
            var functionAddress = GetProcAddress(hModule, functionName);
            if (functionAddress == IntPtr.Zero)
            {
                throw new EntryPointNotFoundException($"Function {functionName} not found in {dllPath}");
            }
            // Return delegate, casting is hack-ish, but simplifies usage
            return (T)(object)(Marshal.GetDelegateForFunctionPointer(functionAddress, typeof(T)));
        }
    }
}
