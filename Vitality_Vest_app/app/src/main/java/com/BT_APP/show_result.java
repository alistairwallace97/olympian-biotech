package com.BT_APP;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import com.dropbox.core.DbxDownloader;
import com.dropbox.core.DbxException;
import com.dropbox.core.DbxRequestConfig;
import com.dropbox.core.v2.DbxClientV2;
import com.dropbox.core.v2.files.FileMetadata;
import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.components.Description;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.components.YAxis;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet;
import java.io.IOException;
import java.io.*;
import java.util.ArrayList;
import static com.github.mikephil.charting.components.XAxis.XAxisPosition.BOTTOM;

public

class show_result extends Activity {

    private ListView listView2;
    private ArrayAdapter<String> listAdapter2;


    @Override
    protected void onCreate(Bundle savedInstanceSate) {
        super.onCreate(savedInstanceSate);
        setContentView(R.layout.result_view);
        final String path = Environment.getExternalStorageDirectory().getAbsolutePath()+"/test/";
        final String ACCESS_TOKEN = "ulsAbO-JC8AAAAAAAAAAIj0q7rYMktruYTB_4iLnzNEipHH7rPgoWggLXCLBNfGn";
        DbxRequestConfig config = new DbxRequestConfig("Dropbox/olympian-biotech");
        DbxClientV2 client = new DbxClientV2(config, ACCESS_TOKEN);
        int sum=0;

        LineChart lineChart1 =  findViewById(R.id.linechart1);
        LineChart lineChart2 = findViewById(R.id.linechart2);
        ArrayList<Entry> EMG = new ArrayList<>();
        ArrayList<Entry> SWITCH = new ArrayList<>();
        ArrayList<Entry> PREDICTED1 = new ArrayList<>();
        ArrayList<Entry> PREDICTED2 = new ArrayList<>();
        ArrayList<Entry> MOTION = new ArrayList<>();
        ArrayList<Entry> TEMPERATURE = new ArrayList<>();
        ArrayList<Entry> SLEEP = new ArrayList<>();


        listView2 = findViewById(R.id.listMessage2);
        listAdapter2 = new ArrayAdapter(this, R.layout.message_detail);
        listView2.setAdapter(listAdapter2);
        listView2.setDivider(null);

        try {
            DbxDownloader<FileMetadata> downloader1 = client.files().download("/graph/graph_test.txt");
            FileOutputStream out1 = new FileOutputStream(path + "/received1.txt/");
            downloader1.download(out1);
            out1.close();

            DbxDownloader<FileMetadata> downloader2 = client.files().download("/graph/all_graphs/graph_all_test.txt");
            FileOutputStream out2 = new FileOutputStream(path + "/received2.txt/");
            downloader2.download(out2);
            out2.close();
        }catch (IOException e1){
            e1.printStackTrace();
        }catch (DbxException e2){
            e2.printStackTrace();
        }
        File read1 = new  File(path+"/received1.txt/");
        String[] graph1 = FileReadWrite.Load(read1);
        File read2 = new  File(path+"/received2.txt/");
        String[] graph2 = FileReadWrite.Load(read2);


        Float x = 0f;

        for (int i = 1;i <graph1.length;i++){
            String[] parts = graph1[i].split(",");
            EMG.add(new Entry(x,Float.valueOf(parts[0])));
            SWITCH.add(new Entry(x,Float.valueOf(parts[1])*5));
            PREDICTED1.add(new Entry(x,Float.valueOf(parts[2])*5));;

            x = x+1f/18f;
        }

        x = 0f;

        for (int i = 0;i<graph2.length;i++){
            if(graph2[i].contains("T")){
                PREDICTED2.add(new Entry(x,0));
                MOTION.add((new Entry(x,0)));
                SLEEP.add((new Entry(x,0)));
                TEMPERATURE.add((new Entry(x,0)));
                x = Float.valueOf(graph2[i].split(":")[1])+Float.valueOf(graph2[i].split(":")[2])/60+Float.valueOf(graph2[i].split(":")[3])/3600;;
            }else if(graph2[i].contains("n")){
                sum = sum + Integer.valueOf(graph2[i].split("n")[1].split(",")[0]);
            }else if(graph2[i].contains("0")){
                String[] parts = graph2[i].split(",");
                PREDICTED2.add(new Entry(x,Float.valueOf(parts[2])*5));
                MOTION.add((new Entry(x,Float.valueOf(parts[3])*10)));
                SLEEP.add((new Entry(x,Float.valueOf(parts[4])*10)));
                TEMPERATURE.add((new Entry(x,0)));
                x = x+1f/18f/3600f;

            }
        }

        ArrayList<ILineDataSet>  lineDataSets1 = new ArrayList<>();
        ArrayList<ILineDataSet>  lineDataSets2 = new ArrayList<>();

        LineDataSet lineDataSet1 = new LineDataSet(EMG,"EMG Data");
        lineDataSet1.setDrawCircles(false);
        lineDataSet1.setColor(getResources().getColor(R.color.black));

        LineDataSet lineDataSet2 = new LineDataSet(SWITCH,"Switch Value");
        lineDataSet2.setDrawCircles(false);
        lineDataSet2.setColor(getResources().getColor(R.color.colorPrimary));

        LineDataSet lineDataSet3 = new LineDataSet(PREDICTED1,"Predicted Result");
        lineDataSet3.setDrawCircles(false);
        lineDataSet3.setDrawFilled(true);
        lineDataSet3.setColor(getResources().getColor(R.color.blue_violet));
        lineDataSet3.setColor(getResources().getColor(R.color.blue_violet));

        LineDataSet lineDataSet4 = new LineDataSet(MOTION,"Motion");
        lineDataSet4.setDrawCircles(false);
        lineDataSet4.setDrawFilled(true);
        lineDataSet4.setColor(getResources().getColor(R.color.pale_violet_red));
        lineDataSet4.setFillColor(getResources().getColor(R.color.pale_violet_red));


        LineDataSet lineDataSet5 = new LineDataSet(TEMPERATURE,"Temperature");
        lineDataSet5.setDrawCircles(false);
        lineDataSet5.setColor(getResources().getColor(R.color.burlyWood));

        LineDataSet lineDataSet6 = new LineDataSet(TEMPERATURE,"Sleeping status");
        lineDataSet6.setDrawCircles(false);
        lineDataSet6.setColor(getResources().getColor(R.color.cold));

        LineDataSet lineDataSet7 = new LineDataSet(PREDICTED2,"Predicted Result");
        lineDataSet7.setDrawCircles(false);
        lineDataSet7.setDrawFilled(true);
        lineDataSet7.setColor(getResources().getColor(R.color.blue_violet));
        lineDataSet7.setColor(getResources().getColor(R.color.blue_violet));


        lineDataSets1.add(lineDataSet1);
        lineDataSets1.add(lineDataSet2);
        lineDataSets1.add(lineDataSet3);
        lineDataSets2.add(lineDataSet6);
        lineDataSets2.add(lineDataSet5);
        lineDataSets2.add(lineDataSet4);
        lineDataSets2.add(lineDataSet7);

        lineChart1.setData(new LineData(lineDataSets1));
        lineChart1.setVisibleYRangeMaximum(16f, YAxis.AxisDependency.LEFT);
        lineChart1.moveViewTo(0,0, YAxis.AxisDependency.LEFT);
        Description des1 = lineChart1.getDescription();
        XAxis xAxis1 = lineChart1.getXAxis();
        des1.setText("received data");
        lineChart1.setDescription(des1);
        xAxis1.setPosition(BOTTOM);
        YAxis yAxis1 = lineChart1.getAxisRight();
        yAxis1.setEnabled(false);
        lineChart1.moveViewTo(0,5, YAxis.AxisDependency.LEFT);

        lineChart2.setData(new LineData(lineDataSets2));
        lineChart2.setVisibleYRangeMaximum(10f, YAxis.AxisDependency.LEFT);
        XAxis xAxis2 = lineChart2.getXAxis();
        Description des2 = lineChart2.getDescription();
        des2.setText("daily record");
        lineChart2.setDescription(des2);
        xAxis2.setAxisMaximum(24);
        xAxis2.setAxisMinimum(0);
        xAxis1.setPosition(BOTTOM);
        YAxis yAxis2 = lineChart2.getAxisRight();
        yAxis2.setEnabled(false);
        lineChart2.setVisibleXRangeMaximum(6f);
        lineChart2.moveViewTo(11,5, YAxis.AxisDependency.LEFT);





        listAdapter2.add("Number of cough detected is \t "+ graph1[0].split(",")[0].split("n")[1]);
        listAdapter2.add("Average Heartrate is \t"+ graph1[0].split(",")[1]);
        listAdapter2.add("Total cough detected is\t" + sum);

    }

}
