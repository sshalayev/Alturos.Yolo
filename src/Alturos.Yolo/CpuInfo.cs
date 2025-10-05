using System;
using System.Collections.Generic;
using System.Text;
using System.Management;

namespace Alturos.Yolo
{
    public static class CpuInfo
    {
        public static string GetModel() 
        {
            string model = "Unknown CPU";
            try
            {
                ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT Name FROM Win32_Processor");

                foreach (ManagementObject queryObj in searcher.Get())
                {
                    model = queryObj["Name"].ToString();
                }
                return model;
            }
            catch (ManagementException e)
            {
                return model;
            }
        }
    }
}
