'''Cell performace analysis and visualization'''
'''Oct 21 Thursday, 9:00PM'''
import pandas as pd
from dask import dataframe as dd
import time 
from datetime import timedelta
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class CellPerformance():
    def __init__(self):
        self.color =['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf',
          '#1a55FF']

    def Read_csv(self,filename):
        """
        This function is used to load input raw data with .csv format

        Parameters
        ----------
        filename : str, the path of input data csv file.

        Returns
        -------
        cell_df : dataframe, load the csv file using Pandas dataframe.

        """
        cell_df = dd.read_csv(filename)
        cell_df = cell_df.compute()
        return cell_df

    def SOC(self, data_df):
        """
        Plot the curve of state of the voltage (V) vs. charge (SOC).

        Parameters
        ----------
        data_df : dataframe, the raw input data load by Read_csv function.

        Returns
        -------
        None.

        """
        unique_cell_id = np.unique(np.array(data_df['Cell_id']))
        for cell_id in unique_cell_id:
            cell_df = data_df[data_df['Cell_id']==cell_id]
            unique_cycle = np.unique(np.array(cell_df['Cycle_Index']))
            plt.subplots()
            soc = np.empty(0)
            i = 0
            colors = iter(cm.rainbow(np.linspace(0, 1, len(unique_cycle)//50)))
            for cycle in unique_cycle:
                if int(cycle)%50 == 0:
                    cycle_df = cell_df[cell_df['Cycle_Index']==cycle]
                    discharge_capacity = np.array(cycle_df['Discharge_Capacity'])
                    soc = discharge_capacity/discharge_capacity.max()
                    voltage = np.array(cycle_df['Voltage'])
                    plt.plot(soc,voltage, label = f'cycle = {cycle}',color = next(colors))
                    i += 1
                else:
                    continue
            plt.xlabel('SOC')
            plt.ylabel('Voltage (V)')
            plt.legend(loc = 'lower left')
            plt.title('SOC')
            plt.show()
        return
    
    def DischargeCapacity_Cycle(self, data_df):
        """
        Plot the curve of discharge capaticy at each cycle.

        Parameters
        ----------
        data_df : dataframe, the raw input data load by Read_csv function.

        Returns
        -------
        None.

        """
        plt.figure()
        unique_cell_id = np.unique(np.array(data_df['Cell_id']))
        i = 0
        for cell_id in unique_cell_id:
            cell_df = data_df[data_df['Cell_id']==cell_id]
            unique_cycle = np.unique(np.array(cell_df['Cycle_Index']))
            discharge_capacity = np.empty(0)
            for cycle in unique_cycle:
                cycle_df = cell_df[cell_df['Cycle_Index']==cycle]
                discharge_capacity = np.append(discharge_capacity, cycle_df['Discharge_Capacity'].max())
            plt.plot(unique_cycle,discharge_capacity, label = f'cell {cell_id}')
            
            # import pdb; pdb.set_trace()
            #Applying a linear fit with .polyfit()
            fit_model = np.poly1d(np.polyfit(unique_cycle, discharge_capacity, 3))
            pred = fit_model(unique_cycle)
            plt.plot(unique_cycle, pred, label = f'Pred cell {cell_id}')
            i += 1
        plt.xlabel('Cycle Number')
        plt.ylabel('Discharge Capacity (Ah)')
        plt.legend() 
        plt.show()
        plt.title('Discharge capacity change with cycle')
        return

    def DischargeCapacity_Temp(self,data_df):
        """
        Plot the discharge capacity at given temperature in each cycle.

        Parameters
        ----------
        data_df : dataframe, the raw input data load by Read_csv function.

        Returns
        -------
        None.

        """
        unique_cell_id = np.unique(np.array(data_df['Cell_id']))
        for cell_id in unique_cell_id:
            cell_df = data_df[data_df['Cell_id']==cell_id]
            unique_cycle = np.unique(np.array(cell_df['Cycle_Index']))
            # plt.subplots()
            discharge_capacity = np.empty(0)
            temp = np.empty(0)
            capacity_loss = np.empty(0)
            loss_rate = np.empty(0)
            loss_rate_temp = np.empty(0)
            max_temp = np.empty(0)
            for cycle in unique_cycle:
                cycle_df = cell_df[cell_df['Cycle_Index']==cycle]
                if cycle ==1.0:
                    ref_cap = cycle_df['Discharge_Capacity'].max()
                discharge_capacity = np.append(discharge_capacity, cycle_df['Discharge_Capacity'].max())
                temp = np.append(temp,cycle_df['Temperature'][cycle_df['Discharge_Capacity'].idxmax()])
                capacity_loss = np.append(capacity_loss,abs(cycle_df['Discharge_Capacity'].max()-ref_cap)/ref_cap*100)
                max_temp = np.append(max_temp,cycle_df['Temperature'].max())
                
                if int(cycle)%50 == 0:
                    loss_rate_temp = np.append(loss_rate_temp,cycle_df['Temperature'][cycle_df['Discharge_Capacity'].idxmax()])
                    loss_rate = np.append(loss_rate,abs(cycle_df['Discharge_Capacity'].max()-ref_cap)/cycle)
                    plt.scatter(np.array(cycle_df['Temperature']),np.array(cycle_df['Discharge_Capacity']),label = f'cycle = {cycle}',s =8)
            plt.xlabel('Temperature')
            plt.ylabel('Discharge Capacity (Ah)')
            plt.legend(loc = 'best')
            plt.title('Discharge capacity change with temperature within each cycle')
            plt.show()

            plt.scatter(temp,discharge_capacity,label = f'cell {cell_id}',s =10)
            plt.xlabel('Temperature')
            plt.ylabel('Discharge Capacity (Ah)')
            plt.legend(loc = 'best')
            plt.title('Discharge capacity change with temperature')
            plt.show()

            plt.plot(temp,capacity_loss,label = f'cell {cell_id}')
            plt.xlabel('Temperature')
            plt.ylabel('Discharge Capacity Loss (%)')
            plt.legend(loc = 'best')
            plt.title('Discharge capacity loss change with temperature')
            plt.show()

            plt.plot(loss_rate_temp,loss_rate,label = f'cell {cell_id}')
            plt.xlabel('Temperature')
            plt.ylabel('Capacity Loss Rate')
            plt.legend(loc = 'best')
            plt.title('Capacity loss rate change with temperature')
            plt.show()

            plt.plot(unique_cycle,max_temp,label = f'cell {cell_id}')
            plt.xlabel('Cycle Number')
            plt.ylabel('Maximum Temperature Each Cycle (degree C)')
            plt.legend(loc = 'best')
            plt.title('Maximum temperature change with cycle')
            plt.show() 
        return

    def IR_Temp(self, data_df):
        """
        Plot the internal resistance at give temperature in each cycle.

        Parameters
        ----------
        data_df : dataframe, the raw input data load by Read_csv function.

        Returns
        -------
        None.

        """
        # import pdb; pdb.set_trace()
        unique_cell_id = np.unique(np.array(data_df['Cell_id']))
        for cell_id in unique_cell_id:
            cell_df = data_df[data_df['Cell_id']==cell_id]
            unique_cycle = np.unique(np.array(cell_df['Cycle_Index']))
            plt.figure()
            plt.xlabel('Temperature')
            plt.ylabel('Internal Resistance')
            IR_cycle = np.empty(0)
            for cycle in unique_cycle:
                cycle_df = cell_df[cell_df['Cycle_Index']==cycle]
                IR_cycle = np.append(IR_cycle,cycle_df['Internal_Resistance'].mean())
                if cycle == 1.0 or cycle%50 == 0:
                    IR = cycle_df['Internal_Resistance']
                    temp = cycle_df['Temperature']
                    plt.scatter(temp, IR, label=f'cycle = {cycle}')
            plt.title('Internal Resistance vs. Temperature')
            plt.legend()
            plt.show()

            plt.plot(unique_cycle,IR_cycle, label = f'cell {cell_id}')
            plt.xlabel('Cycle Number')
            plt.ylabel('Internal Resistance (Ohm)')
            plt.legend(loc = 'best')
            plt.title('Internal Resistance vs. Cycle')
            plt.show()

    def Process(self,input_path):
        plt.close("all")
        start_time = time.time()
        data_df = pd.DataFrame()
        for filename in glob.glob(os.path.join(input_path,'*.csv')):
            cell_df = self.Read_csv(filename)
            discharge = cell_df[cell_df['Current']<0]
            data_df = data_df.append(discharge, ignore_index = True)
        Discharge_soc = self.SOC(data_df)
        DischargeCapacity_Cycle = self.DischargeCapacity_Cycle(data_df)
        DischargeCapacity_Temp = self.DischargeCapacity_Temp(data_df)
        InternalResistance = self.IR_Temp(data_df)

        elapsed_time = int(time.time()-start_time)
        hrs_mins_sec = str(timedelta(seconds = elapsed_time))
        print(f'Cell performance analysis processing time: {hrs_mins_sec}')
        

if __name__ == '__main__':
    input_path = r"./data"
    CellPerformance = CellPerformance()
    CellPerformance.Process(input_path)