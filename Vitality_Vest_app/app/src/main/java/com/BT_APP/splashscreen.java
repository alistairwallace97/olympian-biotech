package com.BT_APP;

/*
* Create a splash screen for app
*/

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import android.widget.TextView;

public class splashscreen extends Activity{
    private TextView tv;
    private ImageView iv;

    @Override
    protected void onCreate(Bundle savedInstanceSate){
        super.onCreate(savedInstanceSate);
        setContentView(R.layout.splash);
        tv = findViewById(R.id.tv);
        iv = findViewById(R.id.iv);
        Animation myanim = AnimationUtils.loadAnimation(this, R.anim.animation);
        tv.startAnimation(myanim);
        iv.startAnimation(myanim);
        final Intent tohomepage = new Intent(this, com.BT_APP.MainActivity.class);
        Thread timer = new Thread(){
            public void run() {
                try {
                    sleep(3000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    startActivity(tohomepage);
                    finish();
                }
            }
        };
            timer.start();
    }
}
